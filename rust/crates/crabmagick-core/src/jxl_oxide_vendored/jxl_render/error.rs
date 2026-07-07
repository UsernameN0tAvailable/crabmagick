#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    Bitstream(crate::jxl_oxide_vendored::jxl_bitstream::Error),
    Decoder(crate::jxl_oxide_vendored::jxl_coding::Error),
    Buffer(crate::jxl_oxide_vendored::jxl_grid::OutOfMemory),
    Modular(crate::jxl_oxide_vendored::jxl_modular::Error),
    Frame(crate::jxl_oxide_vendored::jxl_frame::Error),
    Color(crate::jxl_oxide_vendored::jxl_color::Error),
    IncompleteFrame,
    FailedReference,
    UninitializedLfFrame(u32),
    InvalidReference(u32),
    NotReady,
    NotSupported(&'static str),
}

impl From<crate::jxl_oxide_vendored::jxl_bitstream::Error> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_bitstream::Error) -> Self {
        Self::Bitstream(err)
    }
}

impl From<crate::jxl_oxide_vendored::jxl_coding::Error> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_coding::Error) -> Self {
        Self::Decoder(err)
    }
}

impl From<crate::jxl_oxide_vendored::jxl_grid::OutOfMemory> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_grid::OutOfMemory) -> Self {
        Self::Buffer(err)
    }
}

impl From<crate::jxl_oxide_vendored::jxl_modular::Error> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_modular::Error) -> Self {
        Self::Modular(err)
    }
}

impl From<crate::jxl_oxide_vendored::jxl_frame::Error> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_frame::Error) -> Self {
        Self::Frame(err)
    }
}

impl From<crate::jxl_oxide_vendored::jxl_color::Error> for Error {
    fn from(err: crate::jxl_oxide_vendored::jxl_color::Error) -> Self {
        Self::Color(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Error::*;

        match self {
            Bitstream(err) => write!(f, "bitstream error: {err}"),
            Decoder(err) => write!(f, "entropy decoder error: {err}"),
            Buffer(err) => write!(f, "{err}"),
            Modular(err) => write!(f, "modular subimage decode error: {err}"),
            Frame(err) => write!(f, "frame error: {err}"),
            Color(err) => write!(f, "color management error: {err}"),
            IncompleteFrame => write!(f, "frame data is incomplete"),
            FailedReference => write!(f, "reference frame failed to render"),
            UninitializedLfFrame(lf_level) => {
                write!(f, "uninitialized LF frame for level {lf_level}")
            }
            InvalidReference(idx) => write!(f, "invalid reference {idx}"),
            NotReady => write!(f, "image is not ready to be rendered"),
            NotSupported(msg) => write!(f, "not supported: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        use Error::*;

        match self {
            Bitstream(err) => Some(err),
            Decoder(err) => Some(err),
            Buffer(err) => Some(err),
            Modular(err) => Some(err),
            Frame(err) => Some(err),
            Color(err) => Some(err),
            _ => None,
        }
    }
}

impl Error {
    pub fn unexpected_eof(&self) -> bool {
        match self {
            Error::Bitstream(e) => e.unexpected_eof(),
            Error::Decoder(e) => e.unexpected_eof(),
            Error::Modular(e) => e.unexpected_eof(),
            Error::Frame(e) => e.unexpected_eof(),
            Error::Color(e) => e.unexpected_eof(),
            _ => false,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
