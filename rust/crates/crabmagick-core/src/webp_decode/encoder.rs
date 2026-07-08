//! Encoding of WebP images.
use std::collections::BinaryHeap;
use std::io::{self, Write};

use quick_error::quick_error;

/// Color type of the image.
///
/// Note that the WebP format doesn't have a concept of color type. All images are encoded as RGBA
/// and some decoders may treat them as such. This enum is used to indicate the color type of the
/// input data provided to the encoder, which can help improve compression ratio.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ColorType {
    /// Opaque image with a single luminance byte per pixel.
    L8,
    /// Image with a luminance and alpha byte per pixel.
    La8,
    /// Opaque image with a red, green, and blue byte per pixel.
    Rgb8,
    /// Image with a red, green, blue, and alpha byte per pixel.
    Rgba8,
}

quick_error! {
    /// Error that can occur during encoding.
    #[derive(Debug)]
    #[non_exhaustive]
    pub enum EncodingError {
        /// An IO error occurred.
        IoError(err: io::Error) {
            from()
            display("IO error: {}", err)
            source(err)
        }

        /// The image dimensions are not allowed by the WebP format.
        InvalidDimensions {
            display("Invalid dimensions")
        }
    }
}

struct BitWriter<W> {
    writer: W,
    buffer: u64,
    nbits: u8,
}

impl<W: Write> BitWriter<W> {
    fn write_bits(&mut self, bits: u64, nbits: u8) -> io::Result<()> {
        debug_assert!(nbits <= 64);

        self.buffer |= bits << self.nbits;
        self.nbits += nbits;

        if self.nbits >= 64 {
            self.writer.write_all(&self.buffer.to_le_bytes())?;
            self.nbits -= 64;
            self.buffer = bits.checked_shr(u32::from(nbits - self.nbits)).unwrap_or(0);
        }
        debug_assert!(self.nbits < 64);
        Ok(())
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.nbits % 8 != 0 {
            self.write_bits(0, 8 - self.nbits % 8)?;
        }
        if self.nbits > 0 {
            self.writer
                .write_all(&self.buffer.to_le_bytes()[..self.nbits as usize / 8])
                .unwrap();
            self.buffer = 0;
            self.nbits = 0;
        }
        Ok(())
    }
}

const NUM_LENGTH_CODES: usize = 24;
const GREEN_COPY_CODES: usize = 256 + NUM_LENGTH_CODES;
const DIST_ALPHABET_SIZE: usize = 40;
const HASH_BITS: usize = 18;
const HASH_SIZE: usize = 1 << HASH_BITS;
const MAX_MATCH_LEN: usize = 4096;
const MAX_CHAIN_DEPTH: usize = 32;
const MAX_PLANE_CODE: usize = 1_048_576;
const MAX_BACKWARD_DISTANCE: usize = MAX_PLANE_CODE - 120;

#[rustfmt::skip]
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),  (1, 0),  (1, 1),  (-1, 1), (0, 2),  (2, 0),  (1, 2),  (-1, 2),
    (2, 1),  (-2, 1), (2, 2),  (-2, 2), (0, 3),  (3, 0),  (1, 3),  (-1, 3),
    (3, 1),  (-3, 1), (2, 3),  (-2, 3), (3, 2),  (-3, 2), (0, 4),  (4, 0),
    (1, 4),  (-1, 4), (4, 1),  (-4, 1), (3, 3),  (-3, 3), (2, 4),  (-2, 4),
    (4, 2),  (-4, 2), (0, 5),  (3, 4),  (-3, 4), (4, 3),  (-4, 3), (5, 0),
    (1, 5),  (-1, 5), (5, 1),  (-5, 1), (2, 5),  (-2, 5), (5, 2),  (-5, 2),
    (4, 4),  (-4, 4), (3, 5),  (-3, 5), (5, 3),  (-5, 3), (0, 6),  (6, 0),
    (1, 6),  (-1, 6), (6, 1),  (-6, 1), (2, 6),  (-2, 6), (6, 2),  (-6, 2),
    (4, 5),  (-4, 5), (5, 4),  (-5, 4), (3, 6),  (-3, 6), (6, 3),  (-6, 3),
    (0, 7),  (7, 0),  (1, 7),  (-1, 7), (5, 5),  (-5, 5), (7, 1),  (-7, 1),
    (4, 6),  (-4, 6), (6, 4),  (-6, 4), (2, 7),  (-2, 7), (7, 2),  (-7, 2),
    (3, 7),  (-3, 7), (7, 3),  (-7, 3), (5, 6),  (-5, 6), (6, 5),  (-6, 5),
    (8, 0),  (4, 7),  (-4, 7), (7, 4),  (-7, 4), (8, 1),  (8, 2),  (6, 6),
    (-6, 6), (8, 3),  (5, 7),  (-5, 7), (7, 5),  (-7, 5), (8, 4),  (6, 7),
    (-6, 7), (7, 6),  (-7, 6), (8, 5),  (7, 7),  (-7, 7), (8, 6),  (8, 7)
];

#[derive(Clone, Debug)]
enum Token {
    Literal(u32),
    CacheHit(u16),
    Copy { len: u16, dist: u32 },
}

struct ColorCache {
    table: Vec<u32>,
    bits: u8,
}

impl ColorCache {
    fn new(bits: u8) -> Self {
        Self {
            table: vec![0; 1 << bits],
            bits,
        }
    }

    #[inline(always)]
    fn lookup(&self, argb: u32) -> Option<u16> {
        let idx = self.hash(argb);
        (self.table[idx] == argb).then_some(idx as u16)
    }

    #[inline(always)]
    fn insert(&mut self, argb: u32) {
        let idx = self.hash(argb);
        self.table[idx] = argb;
    }

    #[inline(always)]
    fn hash(&self, argb: u32) -> usize {
        (argb.wrapping_mul(0x1e35a7bd) >> (32 - self.bits)) as usize
    }
}

struct HashChain {
    chain: Vec<i32>,
}

impl HashChain {
    fn build(pixels: &[u32]) -> Self {
        let mut chain = vec![-1; pixels.len()];
        let mut head = vec![-1; HASH_SIZE];
        for (pos, &pixel) in pixels.iter().enumerate() {
            let hash = pixel_hash(pixel);
            chain[pos] = head[hash];
            head[hash] = pos as i32;
        }
        Self { chain }
    }
}

fn write_single_entry_huffman_tree<W: Write>(w: &mut BitWriter<W>, symbol: u16) -> io::Result<()> {
    debug_assert!(symbol < 256);
    w.write_bits(1, 2)?;
    if symbol <= 1 {
        w.write_bits(0, 1)?;
        w.write_bits(u64::from(symbol), 1)?;
    } else {
        w.write_bits(1, 1)?;
        w.write_bits(u64::from(symbol), 8)?;
    }
    Ok(())
}

fn build_huffman_tree(
    frequencies: &[u32],
    lengths: &mut [u8],
    codes: &mut [u16],
    length_limit: u8,
) -> bool {
    assert_eq!(frequencies.len(), lengths.len());
    assert_eq!(frequencies.len(), codes.len());

    if frequencies.iter().filter(|&&f| f > 0).count() <= 1 {
        lengths.fill(0);
        codes.fill(0);
        return false;
    }

    #[derive(Eq, PartialEq, Copy, Clone, Debug)]
    struct Item(u32, u16);
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            other.0.cmp(&self.0)
        }
    }
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    // Build a huffman tree
    let mut internal_nodes = Vec::new();
    let mut nodes = BinaryHeap::from_iter(
        frequencies
            .iter()
            .enumerate()
            .filter(|&(_, &frequency)| frequency > 0)
            .map(|(i, &frequency)| Item(frequency, i as u16)),
    );
    while nodes.len() > 1 {
        let Item(frequency1, index1) = nodes.pop().unwrap();
        let mut root = nodes.peek_mut().unwrap();
        internal_nodes.push((index1, root.1));
        *root = Item(
            frequency1 + root.0,
            internal_nodes.len() as u16 + frequencies.len() as u16 - 1,
        );
    }

    // Walk the tree to assign code lengths
    lengths.fill(0);
    let mut stack = Vec::new();
    stack.push((nodes.pop().unwrap().1, 0));
    while let Some((node, depth)) = stack.pop() {
        let node = node as usize;
        if node < frequencies.len() {
            lengths[node] = depth as u8;
        } else {
            let (left, right) = internal_nodes[node - frequencies.len()];
            stack.push((left, depth + 1));
            stack.push((right, depth + 1));
        }
    }

    // Limit the codes to length length_limit
    let mut max_length = 0;
    for &length in lengths.iter() {
        max_length = max_length.max(length);
    }
    if max_length > length_limit {
        let mut counts = [0u32; 16];
        for &length in lengths.iter() {
            counts[length.min(length_limit) as usize] += 1;
        }

        let mut total = 0;
        for (i, count) in counts
            .iter()
            .enumerate()
            .skip(1)
            .take(length_limit as usize)
        {
            total += count << (length_limit as usize - i);
        }

        while total > 1u32 << length_limit {
            let mut i = length_limit as usize - 1;
            while counts[i] == 0 {
                i -= 1;
            }
            counts[i] -= 1;
            counts[length_limit as usize] -= 1;
            counts[i + 1] += 2;
            total -= 1;
        }

        // assign new lengths
        let mut len = length_limit;
        let mut indexes = frequencies.iter().copied().enumerate().collect::<Vec<_>>();
        indexes.sort_unstable_by_key(|&(_, frequency)| frequency);
        for &(i, frequency) in &indexes {
            if frequency > 0 {
                while counts[len as usize] == 0 {
                    len -= 1;
                }
                lengths[i] = len;
                counts[len as usize] -= 1;
            }
        }
    }

    // Assign codes
    codes.fill(0);
    let mut code = 0u32;
    for len in 1..=length_limit {
        for (i, &length) in lengths.iter().enumerate() {
            if length == len {
                codes[i] = (code as u16).reverse_bits() >> (16 - len);
                code += 1;
            }
        }
        code <<= 1;
    }
    assert_eq!(code, 2 << length_limit);

    true
}

fn write_huffman_tree<W: Write>(
    w: &mut BitWriter<W>,
    frequencies: &[u32],
    lengths: &mut [u8],
    codes: &mut [u16],
) -> io::Result<()> {
    let nonzero = frequencies
        .iter()
        .enumerate()
        .filter_map(|(symbol, &frequency)| (frequency > 0).then_some(symbol))
        .collect::<Vec<_>>();

    if nonzero.is_empty() {
        lengths.fill(0);
        codes.fill(0);
        return write_single_entry_huffman_tree(w, 0);
    }

    if nonzero.len() == 1 && nonzero[0] < 256 {
        lengths.fill(0);
        codes.fill(0);
        return write_single_entry_huffman_tree(w, nonzero[0] as u16);
    }

    if !build_huffman_tree(frequencies, lengths, codes, 15) {
        let mut scratch = frequencies.to_vec();
        let symbol = nonzero[0];
        let dummy = usize::from(symbol == 0);
        scratch[dummy] = scratch[dummy].max(1);
        build_huffman_tree(&scratch, lengths, codes, 15);
    }

    let mut code_length_lengths = [0u8; 16];
    let mut code_length_codes = [0u16; 16];
    let mut code_length_frequencies = [0u32; 16];
    for &length in lengths.iter() {
        code_length_frequencies[length as usize] += 1;
    }
    let single_code_length_length = !build_huffman_tree(
        &code_length_frequencies,
        &mut code_length_lengths,
        &mut code_length_codes,
        7,
    );

    const CODE_LENGTH_ORDER: [usize; 19] = [
        17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    ];

    // Write the huffman tree
    w.write_bits(0, 1)?; // normal huffman tree
    w.write_bits(19 - 4, 4)?; // num_code_lengths - 4

    for i in CODE_LENGTH_ORDER {
        if i > 15 || code_length_frequencies[i] == 0 {
            w.write_bits(0, 3)?;
        } else if single_code_length_length {
            w.write_bits(1, 3)?;
        } else {
            w.write_bits(u64::from(code_length_lengths[i]), 3)?;
        }
    }

    w.write_bits(0, 1)?; // use full alphabet size

    if !single_code_length_length {
        for &len in lengths.iter() {
            w.write_bits(
                u64::from(code_length_codes[len as usize]),
                code_length_lengths[len as usize],
            )?;
        }
    }

    Ok(())
}

#[inline(always)]
fn pixel_hash(pixel: u32) -> usize {
    (pixel.wrapping_mul(0x1e35a7bd) as usize) & (HASH_SIZE - 1)
}

#[inline(always)]
fn pack_pixel(pixel: &[u8]) -> u32 {
    (u32::from(pixel[3]) << 24)
        | (u32::from(pixel[0]) << 16)
        | (u32::from(pixel[1]) << 8)
        | u32::from(pixel[2])
}

#[inline(always)]
fn pixel_channels(pixel: u32) -> (usize, usize, usize, usize) {
    (
        ((pixel >> 16) & 0xFF) as usize,
        ((pixel >> 8) & 0xFF) as usize,
        (pixel & 0xFF) as usize,
        ((pixel >> 24) & 0xFF) as usize,
    )
}

#[inline(always)]
fn value_to_prefix(value: u32) -> (u16, u8, u32) {
    debug_assert!(value >= 1);
    if value <= 4 {
        return ((value - 1) as u16, 0, 0);
    }

    let n = value - 1;
    let highest_bit = 31 - n.leading_zeros();
    let extra_bits = highest_bit - 1;
    let second_highest_bit = (n >> extra_bits) & 1;
    let symbol = (2 * highest_bit + second_highest_bit) as u16;
    let offset = (2 + second_highest_bit) << extra_bits;
    (symbol, extra_bits as u8, n - offset)
}

#[inline(always)]
fn plane_code_to_distance(width: u32, plane_code: usize) -> usize {
    if plane_code > 120 {
        plane_code - 120
    } else {
        let (xoffset, yoffset) = DISTANCE_MAP[plane_code - 1];
        let dist = i32::from(xoffset) + i32::from(yoffset) * width as i32;
        dist.max(1) as usize
    }
}

fn build_small_distance_lookup(width: u32) -> Vec<u16> {
    let max_dist = (1..=120)
        .map(|plane_code| plane_code_to_distance(width, plane_code))
        .max()
        .unwrap_or(1);
    let mut lookup = vec![0u16; max_dist + 1];
    for plane_code in 1..=120u16 {
        let dist = plane_code_to_distance(width, plane_code as usize);
        if let Some(entry) = lookup.get_mut(dist) {
            if *entry == 0 || plane_code < *entry {
                *entry = plane_code;
            }
        }
    }
    lookup
}

#[inline(always)]
fn distance_to_plane_code(dist: usize, small_lookup: &[u16]) -> u32 {
    if dist < small_lookup.len() {
        let code = small_lookup[dist];
        if code != 0 {
            return u32::from(code);
        }
    }
    (dist + 120) as u32
}

fn find_longest_match(pixels: &[u32], chain: &HashChain, pos: usize) -> Option<(usize, usize)> {
    if pos + 1 >= pixels.len() {
        return None;
    }

    let max_len = (pixels.len() - pos).min(MAX_MATCH_LEN);
    let mut best_len = 0usize;
    let mut best_dist = 0usize;
    let mut depth = 0usize;
    let mut candidate_index = chain.chain[pos];

    while candidate_index >= 0 && depth < MAX_CHAIN_DEPTH {
        let candidate = candidate_index as usize;
        let dist = pos - candidate;
        if dist > MAX_BACKWARD_DISTANCE {
            break;
        }

        let mut len = 0usize;
        while len < max_len && pixels[candidate + len] == pixels[pos + len] {
            len += 1;
        }

        if len > best_len || (len == best_len && dist < best_dist) {
            best_len = len;
            best_dist = dist;
            if best_len == max_len {
                break;
            }
        }

        candidate_index = chain.chain[candidate];
        depth += 1;
    }

    (best_len >= 2).then_some((best_len, best_dist))
}

#[inline(always)]
fn should_use_copy(len: usize, dist: usize, cache_hit: bool) -> bool {
    len >= 4 || (len == 3 && (!cache_hit || dist <= 128)) || (len == 2 && dist <= 16 && !cache_hit)
}

fn choose_cache_bits(num_pixels: usize) -> u8 {
    if num_pixels >= 1 << 18 {
        10
    } else if num_pixels >= 1 << 14 {
        9
    } else {
        8
    }
}

fn tokenize_pixels(pixels: &[u32], cache_bits: Option<u8>) -> (Vec<Token>, usize) {
    let chain = HashChain::build(pixels);
    let mut tokens = Vec::with_capacity(pixels.len());
    let mut cache = cache_bits.map(ColorCache::new);
    let mut cache_hits = 0usize;
    let mut pos = 0usize;

    while pos < pixels.len() {
        let cache_hit = cache.as_ref().and_then(|c| c.lookup(pixels[pos]));
        let copy = find_longest_match(pixels, &chain, pos);

        if let Some((len, dist)) = copy {
            let use_copy = if pos + 1 < pixels.len() {
                let next_cache_hit = cache.as_ref().and_then(|c| c.lookup(pixels[pos + 1]));
                let next_copy = find_longest_match(pixels, &chain, pos + 1);
                let skip_for_better_next = next_copy
                    .map(|(next_len, next_dist)| {
                        should_use_copy(next_len, next_dist, next_cache_hit.is_some())
                            && next_len > len + 1
                    })
                    .unwrap_or(false);
                !skip_for_better_next && should_use_copy(len, dist, cache_hit.is_some())
            } else {
                should_use_copy(len, dist, cache_hit.is_some())
            };

            if use_copy {
                tokens.push(Token::Copy {
                    len: len as u16,
                    dist: dist as u32,
                });
                if let Some(cache) = cache.as_mut() {
                    for &pixel in &pixels[pos..pos + len] {
                        cache.insert(pixel);
                    }
                }
                pos += len;
                continue;
            }
        }

        if let Some(index) = cache_hit {
            tokens.push(Token::CacheHit(index));
            cache_hits += 1;
        } else {
            tokens.push(Token::Literal(pixels[pos]));
        }

        if let Some(cache) = cache.as_mut() {
            cache.insert(pixels[pos]);
        }
        pos += 1;
    }

    (tokens, cache_hits)
}

fn count_token_frequencies(
    tokens: &[Token],
    color: ColorType,
    width: u32,
    green_alphabet_size: usize,
) -> (
    Vec<u32>,
    [u32; 256],
    [u32; 256],
    [u32; 256],
    [u32; DIST_ALPHABET_SIZE],
) {
    let mut frequencies0 = [0u32; 256];
    let mut frequencies1 = vec![0u32; green_alphabet_size];
    let mut frequencies2 = [0u32; 256];
    let mut frequencies3 = [0u32; 256];
    let mut frequencies_dist = [0u32; DIST_ALPHABET_SIZE];
    let small_lookup = build_small_distance_lookup(width);

    match color {
        ColorType::L8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
            frequencies3[0] = 1;
        }
        ColorType::La8 => {
            frequencies0[0] = 1;
            frequencies2[0] = 1;
        }
        ColorType::Rgb8 => {
            frequencies3[0] = 1;
        }
        ColorType::Rgba8 => {}
    }

    for token in tokens {
        match *token {
            Token::Literal(pixel) => {
                let (red, green, blue, alpha) = pixel_channels(pixel);
                frequencies1[green] += 1;
                match color {
                    ColorType::L8 => {}
                    ColorType::La8 => {
                        frequencies3[alpha] += 1;
                    }
                    ColorType::Rgb8 => {
                        frequencies0[red] += 1;
                        frequencies2[blue] += 1;
                    }
                    ColorType::Rgba8 => {
                        frequencies0[red] += 1;
                        frequencies2[blue] += 1;
                        frequencies3[alpha] += 1;
                    }
                }
            }
            Token::CacheHit(index) => {
                frequencies1[GREEN_COPY_CODES + usize::from(index)] += 1;
            }
            Token::Copy { len, dist } => {
                let (length_symbol, _, _) = value_to_prefix(u32::from(len));
                frequencies1[256 + usize::from(length_symbol)] += 1;

                let plane_code = distance_to_plane_code(dist as usize, &small_lookup);
                let (dist_symbol, _, _) = value_to_prefix(plane_code);
                frequencies_dist[usize::from(dist_symbol)] += 1;
            }
        }
    }

    (
        frequencies1,
        frequencies0,
        frequencies2,
        frequencies3,
        frequencies_dist,
    )
}

fn write_tokens<W: Write>(
    w: &mut BitWriter<W>,
    tokens: &[Token],
    color: ColorType,
    width: u32,
    codes0: &[u16; 256],
    lengths0: &[u8; 256],
    codes1: &[u16],
    lengths1: &[u8],
    codes2: &[u16; 256],
    lengths2: &[u8; 256],
    codes3: &[u16; 256],
    lengths3: &[u8; 256],
    codes_dist: &[u16; DIST_ALPHABET_SIZE],
    lengths_dist: &[u8; DIST_ALPHABET_SIZE],
) -> io::Result<()> {
    let small_lookup = build_small_distance_lookup(width);

    for token in tokens {
        match *token {
            Token::Literal(pixel) => {
                let (red, green, blue, alpha) = pixel_channels(pixel);
                match color {
                    ColorType::L8 => {
                        w.write_bits(u64::from(codes1[green]), lengths1[green])?;
                    }
                    ColorType::La8 => {
                        let len1 = lengths1[green];
                        let len3 = lengths3[alpha];
                        let code = u64::from(codes1[green]) | (u64::from(codes3[alpha]) << len1);
                        w.write_bits(code, len1 + len3)?;
                    }
                    ColorType::Rgb8 => {
                        let len1 = lengths1[green];
                        let len0 = lengths0[red];
                        let len2 = lengths2[blue];
                        let code = u64::from(codes1[green])
                            | (u64::from(codes0[red]) << len1)
                            | (u64::from(codes2[blue]) << (len1 + len0));
                        w.write_bits(code, len1 + len0 + len2)?;
                    }
                    ColorType::Rgba8 => {
                        let len1 = lengths1[green];
                        let len0 = lengths0[red];
                        let len2 = lengths2[blue];
                        let len3 = lengths3[alpha];
                        let code = u64::from(codes1[green])
                            | (u64::from(codes0[red]) << len1)
                            | (u64::from(codes2[blue]) << (len1 + len0))
                            | (u64::from(codes3[alpha]) << (len1 + len0 + len2));
                        w.write_bits(code, len1 + len0 + len2 + len3)?;
                    }
                }
            }
            Token::CacheHit(index) => {
                let symbol = GREEN_COPY_CODES + usize::from(index);
                w.write_bits(u64::from(codes1[symbol]), lengths1[symbol])?;
            }
            Token::Copy { len, dist } => {
                let (length_symbol, length_extra_bits, length_extra_value) =
                    value_to_prefix(u32::from(len));
                let green_symbol = 256 + usize::from(length_symbol);
                w.write_bits(u64::from(codes1[green_symbol]), lengths1[green_symbol])?;
                w.write_bits(u64::from(length_extra_value), length_extra_bits)?;

                let plane_code = distance_to_plane_code(dist as usize, &small_lookup);
                let (dist_symbol, dist_extra_bits, dist_extra_value) = value_to_prefix(plane_code);
                w.write_bits(
                    u64::from(codes_dist[usize::from(dist_symbol)]),
                    lengths_dist[usize::from(dist_symbol)],
                )?;
                w.write_bits(u64::from(dist_extra_value), dist_extra_bits)?;
            }
        }
    }

    Ok(())
}

/// Allows fine-tuning some encoder parameters.
///
/// Pass to [`WebPEncoder::set_params()`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct EncoderParams {
    /// Use a predictor transform. Enabled by default.
    pub use_predictor_transform: bool,
}

impl Default for EncoderParams {
    fn default() -> Self {
        Self {
            use_predictor_transform: true,
        }
    }
}

/// Encode image data with the indicated color type.
///
/// # Panics
///
/// Panics if the image data is not of the indicated dimensions.
fn encode_frame<W: Write>(
    writer: W,
    data: &[u8],
    width: u32,
    height: u32,
    color: ColorType,
    params: EncoderParams,
) -> Result<(), EncodingError> {
    let w = &mut BitWriter {
        writer,
        buffer: 0,
        nbits: 0,
    };

    let (is_color, is_alpha, bytes_per_pixel) = match color {
        ColorType::L8 => (false, false, 1),
        ColorType::La8 => (false, true, 2),
        ColorType::Rgb8 => (true, false, 3),
        ColorType::Rgba8 => (true, true, 4),
    };

    assert_eq!(
        (u64::from(width) * u64::from(height)).saturating_mul(bytes_per_pixel),
        data.len() as u64
    );

    if width == 0 || width > 16384 || height == 0 || height > 16384 {
        return Err(EncodingError::InvalidDimensions);
    }

    w.write_bits(0x2f, 8)?; // signature
    w.write_bits(u64::from(width) - 1, 14)?;
    w.write_bits(u64::from(height) - 1, 14)?;

    w.write_bits(u64::from(is_alpha), 1)?; // alpha used
    w.write_bits(0x0, 3)?; // version

    // subtract green transform
    w.write_bits(0b101, 3)?;

    // predictor transform
    if params.use_predictor_transform {
        w.write_bits(0b111001, 6)?;
        w.write_bits(0x0, 1)?; // no color cache
        write_single_entry_huffman_tree(w, 2)?;
        for _ in 0..4 {
            write_single_entry_huffman_tree(w, 0)?;
        }
    }

    // transforms done
    w.write_bits(0x0, 1)?;

    // expand to RGBA
    let mut pixels = match color {
        ColorType::L8 => data.iter().flat_map(|&p| [p, p, p, 255]).collect(),
        ColorType::La8 => data
            .chunks_exact(2)
            .flat_map(|p| [p[0], p[0], p[0], p[1]])
            .collect(),
        ColorType::Rgb8 => data
            .chunks_exact(3)
            .flat_map(|p| [p[0], p[1], p[2], 255])
            .collect(),
        ColorType::Rgba8 => data.to_vec(),
    };

    // compute subtract green transform
    for pixel in pixels.chunks_exact_mut(4) {
        pixel[0] = pixel[0].wrapping_sub(pixel[1]);
        pixel[2] = pixel[2].wrapping_sub(pixel[1]);
    }

    // compute predictor transform
    if params.use_predictor_transform {
        let row_bytes = width as usize * 4;
        for y in (1..height as usize).rev() {
            let (prev, current) =
                pixels[(y - 1) * row_bytes..][..row_bytes * 2].split_at_mut(row_bytes);
            for (c, p) in current.iter_mut().zip(prev) {
                *c = c.wrapping_sub(*p);
            }
        }
        for i in (4..row_bytes).rev() {
            pixels[i] = pixels[i].wrapping_sub(pixels[i - 4]);
        }
        pixels[3] = pixels[3].wrapping_sub(255);
    }

    let transformed = pixels.chunks_exact(4).map(pack_pixel).collect::<Vec<_>>();

    let suggested_cache_bits = choose_cache_bits(transformed.len());
    let (mut tokens, cache_hits) = tokenize_pixels(&transformed, Some(suggested_cache_bits));
    let cache_bits = if cache_hits == 0 {
        tokens = tokenize_pixels(&transformed, None).0;
        None
    } else {
        Some(suggested_cache_bits)
    };

    // color cache
    if let Some(bits) = cache_bits {
        w.write_bits(1, 1)?;
        w.write_bits(u64::from(bits), 4)?;
    } else {
        w.write_bits(0, 1)?;
    }

    // meta-huffman codes
    w.write_bits(0x0, 1)?;

    let green_alphabet_size = GREEN_COPY_CODES + cache_bits.map_or(0, |bits| 1usize << bits);
    let (frequencies1, frequencies0, frequencies2, frequencies3, frequencies_dist) =
        count_token_frequencies(&tokens, color, width, green_alphabet_size);

    // compute and write huffman codes
    let mut lengths0 = [0u8; 256];
    let mut lengths1 = vec![0u8; green_alphabet_size];
    let mut lengths2 = [0u8; 256];
    let mut lengths3 = [0u8; 256];
    let mut lengths_dist = [0u8; DIST_ALPHABET_SIZE];
    let mut codes0 = [0u16; 256];
    let mut codes1 = vec![0u16; green_alphabet_size];
    let mut codes2 = [0u16; 256];
    let mut codes3 = [0u16; 256];
    let mut codes_dist = [0u16; DIST_ALPHABET_SIZE];
    write_huffman_tree(w, &frequencies1, &mut lengths1, &mut codes1)?;
    if is_color {
        write_huffman_tree(w, &frequencies0, &mut lengths0, &mut codes0)?;
        write_huffman_tree(w, &frequencies2, &mut lengths2, &mut codes2)?;
    } else {
        write_single_entry_huffman_tree(w, 0)?;
        write_single_entry_huffman_tree(w, 0)?;
    }
    if is_alpha {
        write_huffman_tree(w, &frequencies3, &mut lengths3, &mut codes3)?;
    } else if params.use_predictor_transform {
        write_single_entry_huffman_tree(w, 0)?;
    } else {
        write_single_entry_huffman_tree(w, 255)?;
    }
    write_huffman_tree(w, &frequencies_dist, &mut lengths_dist, &mut codes_dist)?;

    // Write image data
    write_tokens(
        w,
        &tokens,
        color,
        width,
        &codes0,
        &lengths0,
        &codes1,
        &lengths1,
        &codes2,
        &lengths2,
        &codes3,
        &lengths3,
        &codes_dist,
        &lengths_dist,
    )?;

    w.flush()?;
    Ok(())
}

const fn chunk_size(inner_bytes: usize) -> u32 {
    if inner_bytes % 2 == 1 {
        (inner_bytes + 1) as u32 + 8
    } else {
        inner_bytes as u32 + 8
    }
}

fn write_chunk<W: Write>(mut w: W, name: &[u8], data: &[u8]) -> io::Result<()> {
    debug_assert!(name.len() == 4);

    w.write_all(name)?;
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    w.write_all(data)?;
    if data.len() % 2 == 1 {
        w.write_all(&[0])?;
    }
    Ok(())
}

/// WebP Encoder.
pub struct WebPEncoder<W> {
    writer: W,
    icc_profile: Vec<u8>,
    exif_metadata: Vec<u8>,
    xmp_metadata: Vec<u8>,
    params: EncoderParams,
}

impl<W: Write> WebPEncoder<W> {
    /// Create a new encoder that writes its output to `w`.
    ///
    /// Only supports "VP8L" lossless encoding.
    pub fn new(w: W) -> Self {
        Self {
            writer: w,
            icc_profile: Vec::new(),
            exif_metadata: Vec::new(),
            xmp_metadata: Vec::new(),
            params: EncoderParams::default(),
        }
    }

    /// Set the ICC profile to use for the image.
    pub fn set_icc_profile(&mut self, icc_profile: Vec<u8>) {
        self.icc_profile = icc_profile;
    }

    /// Set the EXIF metadata to use for the image.
    pub fn set_exif_metadata(&mut self, exif_metadata: Vec<u8>) {
        self.exif_metadata = exif_metadata;
    }

    /// Set the XMP metadata to use for the image.
    pub fn set_xmp_metadata(&mut self, xmp_metadata: Vec<u8>) {
        self.xmp_metadata = xmp_metadata;
    }

    /// Set the `EncoderParams` to use.
    pub fn set_params(&mut self, params: EncoderParams) {
        self.params = params;
    }

    /// Encode image data with the indicated color type.
    ///
    /// # Panics
    ///
    /// Panics if the image data is not of the indicated dimensions.
    pub fn encode(
        mut self,
        data: &[u8],
        width: u32,
        height: u32,
        color: ColorType,
    ) -> Result<(), EncodingError> {
        let mut frame = Vec::new();
        encode_frame(&mut frame, data, width, height, color, self.params)?;

        // If the image has no metadata, it can be encoded with the "simple" WebP container format.
        if self.icc_profile.is_empty()
            && self.exif_metadata.is_empty()
            && self.xmp_metadata.is_empty()
        {
            self.writer.write_all(b"RIFF")?;
            self.writer
                .write_all(&(chunk_size(frame.len()) + 4).to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;
            write_chunk(&mut self.writer, b"VP8L", &frame)?;
        } else {
            let mut total_bytes = 22 + chunk_size(frame.len());
            if !self.icc_profile.is_empty() {
                total_bytes += chunk_size(self.icc_profile.len());
            }
            if !self.exif_metadata.is_empty() {
                total_bytes += chunk_size(self.exif_metadata.len());
            }
            if !self.xmp_metadata.is_empty() {
                total_bytes += chunk_size(self.xmp_metadata.len());
            }

            let mut flags = 0;
            if !self.xmp_metadata.is_empty() {
                flags |= 1 << 2;
            }
            if !self.exif_metadata.is_empty() {
                flags |= 1 << 3;
            }
            if let ColorType::La8 | ColorType::Rgba8 = color {
                flags |= 1 << 4;
            }
            if !self.icc_profile.is_empty() {
                flags |= 1 << 5;
            }

            self.writer.write_all(b"RIFF")?;
            self.writer.write_all(&total_bytes.to_le_bytes())?;
            self.writer.write_all(b"WEBP")?;

            let mut vp8x = Vec::new();
            vp8x.write_all(&[flags])?; // flags
            vp8x.write_all(&[0; 3])?; // reserved
            vp8x.write_all(&(width - 1).to_le_bytes()[..3])?; // canvas width
            vp8x.write_all(&(height - 1).to_le_bytes()[..3])?; // canvas height
            write_chunk(&mut self.writer, b"VP8X", &vp8x)?;

            if !self.icc_profile.is_empty() {
                write_chunk(&mut self.writer, b"ICCP", &self.icc_profile)?;
            }

            write_chunk(&mut self.writer, b"VP8L", &frame)?;

            if !self.exif_metadata.is_empty() {
                write_chunk(&mut self.writer, b"EXIF", &self.exif_metadata)?;
            }

            if !self.xmp_metadata.is_empty() {
                write_chunk(&mut self.writer, b"XMP ", &self.xmp_metadata)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    #[test]
    fn write_webp() {
        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        WebPEncoder::new(&mut output)
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgba8)
            .unwrap();

        let mut decoder =
            crate::webp_decode::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();
        let mut img2 = vec![0; 256 * 256 * 4];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);
    }

    #[test]
    fn write_webp_exif() {
        let mut img = vec![0; 256 * 256 * 3];
        rand::thread_rng().fill_bytes(&mut img);

        let mut exif = vec![0; 10];
        rand::thread_rng().fill_bytes(&mut exif);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_exif_metadata(exif.clone());
        encoder
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgb8)
            .unwrap();

        let mut decoder =
            crate::webp_decode::WebPDecoder::new(std::io::Cursor::new(output)).unwrap();

        let mut img2 = vec![0; 256 * 256 * 3];
        decoder.read_image(&mut img2).unwrap();
        assert_eq!(img, img2);

        let exif2 = decoder.exif_metadata().unwrap();
        assert_eq!(Some(exif), exif2);
    }

    #[test]
    fn roundtrip_libwebp() {
        roundtrip_libwebp_params(EncoderParams::default());
        roundtrip_libwebp_params(EncoderParams {
            use_predictor_transform: false,
            ..Default::default()
        });
    }

    fn roundtrip_libwebp_params(params: EncoderParams) {
        println!("Testing {params:?}");

        let mut img = vec![0; 256 * 256 * 4];
        rand::thread_rng().fill_bytes(&mut img);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(
                &img[..256 * 256 * 3],
                256,
                256,
                crate::webp_decode::ColorType::Rgb8,
            )
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img[..256 * 256 * 3], *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_icc_profile(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params.clone());
        encoder.set_exif_metadata(vec![0; 10]);
        encoder
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);

        let mut output = Vec::new();
        let mut encoder = WebPEncoder::new(&mut output);
        encoder.set_params(params);
        encoder.set_xmp_metadata(vec![0; 7]);
        encoder.set_icc_profile(vec![0; 8]);
        encoder.set_icc_profile(vec![0; 9]);
        encoder
            .encode(&img, 256, 256, crate::webp_decode::ColorType::Rgba8)
            .unwrap();
        let decoded = webp::Decoder::new(&output).decode().unwrap();
        assert_eq!(img, *decoded);
    }
}
