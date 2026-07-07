#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(crate::jxl_decode::jxl_bitstream::Error),
    Decoder(crate::jxl_decode::jxl_coding::Error),
    Buffer(crate::jxl_decode::jxl_grid::OutOfMemory),
    Modular(crate::jxl_decode::jxl_modular::Error),
    VarDct(crate::jxl_decode::jxl_vardct::Error),
    InvalidTocPermutation,
    IncompleteFrameData { field: &'static str },
    OutOfMemory,
    HadError,
}

impl From<crate::jxl_decode::jxl_bitstream::Error> for Error {
    fn from(err: crate::jxl_decode::jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}

impl From<crate::jxl_decode::jxl_coding::Error> for Error {
    fn from(err: crate::jxl_decode::jxl_coding::Error) -> Self {
        Self::Decoder(err)
    }
}

impl From<crate::jxl_decode::jxl_grid::OutOfMemory> for Error {
    fn from(err: crate::jxl_decode::jxl_grid::OutOfMemory) -> Self {
        Self::Buffer(err)
    }
}

impl From<crate::jxl_decode::jxl_modular::Error> for Error {
    fn from(err: crate::jxl_decode::jxl_modular::Error) -> Self {
        Self::Modular(err)
    }
}

impl From<crate::jxl_decode::jxl_vardct::Error> for Error {
    fn from(err: crate::jxl_decode::jxl_vardct::Error) -> Self {
        Self::VarDct(err)
    }
}

impl From<std::collections::TryReserveError> for Error {
    fn from(_: std::collections::TryReserveError) -> Self {
        Self::OutOfMemory
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bitstream(err) => write!(f, "bitstream error: {err}"),
            Self::Decoder(err) => write!(f, "entropy decoder error: {err}"),
            Self::Buffer(err) => write!(f, "{err}"),
            Self::Modular(err) => write!(f, "modular stream error: {err}"),
            Self::VarDct(err) => write!(f, "vardct error: {err}"),
            Self::InvalidTocPermutation => write!(f, "invalid TOC permutation"),
            Self::IncompleteFrameData { field } => {
                write!(f, "incomplete frame data: {field} is missing")
            }
            Self::OutOfMemory => write!(f, "out of memory"),
            Self::HadError => write!(f, "previous parsing errored"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Bitstream(err) => Some(err),
            Self::Decoder(err) => Some(err),
            Self::Buffer(err) => Some(err),
            Self::Modular(err) => Some(err),
            Self::VarDct(err) => Some(err),
            _ => None,
        }
    }
}

impl Error {
    /// Returns whether the error is caused by the unexpected EOF of the bitstream.
    pub fn unexpected_eof(&self) -> bool {
        let bitstream_err = match self {
            Self::Bitstream(b)
            | Self::Decoder(crate::jxl_decode::jxl_coding::Error::Bitstream(b))
            | Self::Modular(crate::jxl_decode::jxl_modular::Error::Decoder(crate::jxl_decode::jxl_coding::Error::Bitstream(b)))
            | Self::Modular(crate::jxl_decode::jxl_modular::Error::Bitstream(b))
            | Self::VarDct(crate::jxl_decode::jxl_vardct::Error::Bitstream(b))
            | Self::VarDct(crate::jxl_decode::jxl_vardct::Error::Decoder(crate::jxl_decode::jxl_coding::Error::Bitstream(b)))
            | Self::VarDct(crate::jxl_decode::jxl_vardct::Error::Modular(crate::jxl_decode::jxl_modular::Error::Bitstream(b)))
            | Self::VarDct(crate::jxl_decode::jxl_vardct::Error::Modular(crate::jxl_decode::jxl_modular::Error::Decoder(
                crate::jxl_decode::jxl_coding::Error::Bitstream(b),
            ))) => b,
            _ => return false,
        };
        if let crate::jxl_decode::jxl_bitstream::Error::Io(e) = bitstream_err {
            e.kind() == std::io::ErrorKind::UnexpectedEof
        } else {
            false
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
