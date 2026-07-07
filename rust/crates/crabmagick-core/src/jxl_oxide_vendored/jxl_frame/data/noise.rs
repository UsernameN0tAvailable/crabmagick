#[derive(Debug)]
pub struct NoiseParameters {
    pub lut: [f32; 8],
}

impl<Ctx> crate::jxl_oxide_vendored::jxl_oxide_common::Bundle<Ctx> for NoiseParameters {
    type Error = crate::jxl_oxide_vendored::jxl_frame::Error;

    fn parse(bitstream: &mut crate::jxl_oxide_vendored::jxl_bitstream::Bitstream, _: Ctx) -> crate::jxl_oxide_vendored::jxl_frame::Result<Self> {
        let mut lut = [0.0f32; 8];
        for slot in &mut lut {
            *slot = bitstream.read_bits(10)? as f32 / (1 << 10) as f32;
        }

        Ok(Self { lut })
    }
}
