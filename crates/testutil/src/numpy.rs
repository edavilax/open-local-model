use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub struct Array<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

pub trait Element: Sized + Copy {
    const DESCR: &str; // "<f4" for f32, "<u4" for u32
    fn from_le_bytes(b: &[u8]) -> Result<Self, std::array::TryFromSliceError>;
}

impl Element for f32 {
    const DESCR: &str = "<f4";
    fn from_le_bytes(b: &[u8]) -> Result<Self, std::array::TryFromSliceError> {
        let bytes: [u8; 4] = b.try_into()?;
        Ok(f32::from_le_bytes(bytes))
    }
}

impl Element for u32 {
    const DESCR: &str = "<u4";
    fn from_le_bytes(b: &[u8]) -> Result<Self, std::array::TryFromSliceError> {
        let bytes: [u8; 4] = b.try_into()?;
        Ok(u32::from_le_bytes(bytes))
    }
}

#[derive(Debug)]
pub enum NpyError {
    Io(std::io::Error),
    Utf8(std::str::Utf8Error),
    FileTooShort(String),
    BadMagic,
    BadHeader(String),
    DtypeMismatch {
        expected: &'static str,
        found: String,
    },
    FortranOrder,
    SizeMismatch {
        expected: usize,
        found: usize,
    },
}

/// Parsed contents of the .npy header dict.
struct Header {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// Pull the value text that follows `'key':` in the header dict.
fn header_value<'a>(header: &'a str, key: &str) -> Result<&'a str, NpyError> {
    let needle = format!("'{key}':");
    let start = header
        .find(&needle)
        .ok_or_else(|| NpyError::BadHeader(format!("missing key {key}")))?
        + needle.len();
    Ok(header[start..].trim_start())
}

/// Parse `{'descr': '<f4', 'fortran_order': False, 'shape': (29, 64), }`.
fn parse_header(header: &str) -> Result<Header, NpyError> {
    // descr: quoted string
    let rest = header_value(header, "descr")?;
    let rest = rest
        .strip_prefix('\'')
        .ok_or_else(|| NpyError::BadHeader("descr is not quoted".into()))?;
    let end = rest
        .find('\'')
        .ok_or_else(|| NpyError::BadHeader("unterminated descr".into()))?;
    let descr = rest[..end].to_string();

    // fortran_order: True | False
    let rest = header_value(header, "fortran_order")?;
    let fortran_order = if rest.starts_with("True") {
        true
    } else if rest.starts_with("False") {
        false
    } else {
        return Err(NpyError::BadHeader(
            "fortran_order is not True/False".into(),
        ));
    };

    // shape: (), (1024,), (29, 64)
    let rest = header_value(header, "shape")?;
    let rest = rest
        .strip_prefix('(')
        .ok_or_else(|| NpyError::BadHeader("shape is not a tuple".into()))?;
    let end = rest
        .find(')')
        .ok_or_else(|| NpyError::BadHeader("unterminated shape".into()))?;
    let shape = rest[..end]
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty()) // trailing comma in (1024,)
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| NpyError::BadHeader(format!("bad shape dimension {s:?}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Header {
        descr,
        fortran_order,
        shape,
    })
}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        NpyError::Io(e)
    }
}

impl From<std::str::Utf8Error> for NpyError {
    fn from(e: std::str::Utf8Error) -> Self {
        NpyError::Utf8(e)
    }
}

impl From<std::array::TryFromSliceError> for NpyError {
    fn from(_: std::array::TryFromSliceError) -> Self {
        NpyError::FileTooShort(String::from("data elem"))
    }
}

pub fn load<T: Element>(path: impl AsRef<Path>) -> Result<Array<T>, NpyError> {
    let fbytes = std::fs::read(path)?;
    let fbuf = fbytes.as_slice();
    // Check magic
    let (magic, fbuf) = fbuf
        .split_at_checked(6)
        .ok_or(NpyError::FileTooShort(String::from("magic")))?;
    if magic != b"\x93NUMPY" {
        return Err(NpyError::BadMagic);
    }
    // Get the major version
    let (vrsn_buf, fbuf) = fbuf
        .split_at_checked(2)
        .ok_or(NpyError::FileTooShort(String::from("version")))?;
    let major_vrsn = vrsn_buf[0];
    if !(1..=3).contains(&major_vrsn) {
        return Err(NpyError::BadHeader(format!(
            "major version must be between 1 and 3, got {major_vrsn}"
        )));
    }
    // Get header length
    let len = if major_vrsn == 1 { 2 } else { 4 };
    let (header_len_buf, fbuf) = fbuf
        .split_at_checked(len)
        .ok_or(NpyError::FileTooShort(String::from("header len")))?;
    let header_len: usize = if major_vrsn == 1 {
        u16::from_le_bytes([header_len_buf[0], header_len_buf[1]]).into()
    } else {
        u32::from_le_bytes([
            header_len_buf[0],
            header_len_buf[1],
            header_len_buf[2],
            header_len_buf[3],
        ]) as usize
    };
    // Get header
    let (header_buf, fbuf) = fbuf
        .split_at_checked(header_len)
        .ok_or(NpyError::FileTooShort(String::from("header")))?;
    let header_str = str::from_utf8(header_buf)?;
    let header = parse_header(header_str)?;
    // Validate header
    if header.descr != T::DESCR {
        return Err(NpyError::DtypeMismatch {
            expected: T::DESCR,
            found: header.descr,
        });
    }
    if header.fortran_order {
        return Err(NpyError::FortranOrder);
    }
    // Check the remaining size supports the data.
    // An empty shape is a 0-d array holding exactly one element, and
    // `product()` of an empty iterator is 1, which gives that for free.
    let data_cnt: usize = header.shape.iter().product();
    let elem_bytes = size_of::<T>();
    let data_size = data_cnt * elem_bytes;
    if fbuf.len() != data_size {
        return Err(NpyError::SizeMismatch {
            expected: data_size,
            found: fbuf.len(),
        });
    }
    // Get the data and return. The length check above guarantees
    // `chunks_exact` yields exactly `data_cnt` chunks with no remainder.
    let data = fbuf
        .chunks_exact(elem_bytes)
        .map(T::from_le_bytes)
        .collect::<Result<Vec<T>, _>>()?;
    Ok(Array {
        shape: header.shape,
        data,
    })
}
