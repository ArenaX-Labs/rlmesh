//! A part's value on the cable is dense: its fields back to back, with no
//! padding (spec §3.3). In the session each field starts on an 8-byte
//! boundary and the stride is rounded up to 8 (session spec §10.6). The
//! bridge converts between the two with the part's field offsets and
//! lengths (`wire::Layout::Packed`); the board only ever sees dense bytes.

/// Why a conversion was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackError {
    /// A buffer isn't the size the layout says.
    Length {
        /// The size the layout needs.
        want: usize,
        /// The size given.
        got: usize,
    },
    /// The offsets and lengths don't describe a layout: different counts,
    /// or a field past the stride.
    Layout,
}

/// The dense size of fields of `lens` bytes.
pub fn dense_len(lens: &[usize]) -> usize {
    lens.iter().sum()
}

fn check(offsets: &[usize], lens: &[usize], stride: usize) -> Result<(), PackError> {
    if offsets.len() != lens.len()
        || offsets
            .iter()
            .zip(lens)
            .any(|(o, l)| o.checked_add(*l).is_none_or(|end| end > stride))
    {
        return Err(PackError::Layout);
    }
    Ok(())
}

fn exact(want: usize, got: usize) -> Result<(), PackError> {
    if want == got {
        Ok(())
    } else {
        Err(PackError::Length { want, got })
    }
}

/// Copies an aligned value (`stride` bytes, fields at `offsets`) into its
/// dense form. `dense` must be exactly [`dense_len`] bytes.
pub fn to_dense(
    aligned: &[u8],
    offsets: &[usize],
    lens: &[usize],
    dense: &mut [u8],
) -> Result<(), PackError> {
    check(offsets, lens, aligned.len())?;
    exact(dense_len(lens), dense.len())?;
    let mut at = 0;
    for (&o, &l) in offsets.iter().zip(lens) {
        dense[at..at + l].copy_from_slice(&aligned[o..o + l]);
        at += l;
    }
    Ok(())
}

/// Spreads a dense value over an aligned buffer of the part's stride,
/// zeroing the padding. `dense` must be exactly [`dense_len`] bytes.
pub fn to_aligned(
    dense: &[u8],
    offsets: &[usize],
    lens: &[usize],
    aligned: &mut [u8],
) -> Result<(), PackError> {
    check(offsets, lens, aligned.len())?;
    exact(dense_len(lens), dense.len())?;
    aligned.fill(0);
    let mut at = 0;
    for (&o, &l) in offsets.iter().zip(lens) {
        aligned[o..o + l].copy_from_slice(&dense[at..at + l]);
        at += l;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_f32_fields_pack_to_eight_bytes_and_back() {
        // `pan: f32`, `tilt: f32`: aligned at 0 and 8, stride 16.
        let (offsets, lens) = ([0, 8], [4, 4]);
        let mut aligned = [0xEE_u8; 16];
        aligned[..4].copy_from_slice(&1.5_f32.to_le_bytes());
        aligned[8..12].copy_from_slice(&(-0.25_f32).to_le_bytes());
        let mut dense = [0_u8; 8];
        to_dense(&aligned, &offsets, &lens, &mut dense).expect("dense");
        assert_eq!(&dense[..4], &1.5_f32.to_le_bytes());
        assert_eq!(&dense[4..], &(-0.25_f32).to_le_bytes());
        let mut back = [0xEE_u8; 16];
        to_aligned(&dense, &offsets, &lens, &mut back).expect("aligned");
        assert_eq!(&back[..4], &aligned[..4]);
        assert_eq!(&back[8..12], &aligned[8..12]);
        assert_eq!(&back[4..8], &[0; 4], "padding is zeroed");
        assert_eq!(&back[12..], &[0; 4]);
    }

    #[test]
    fn mixed_widths_keep_field_order() {
        // `pos: f32[3]` (12 bytes at 0), `on: bool` (1 at 16), `id: u8[2]` (2 at 24): stride 32.
        let (offsets, lens) = ([0, 16, 24], [12, 1, 2]);
        let dense: [u8; 15] = core::array::from_fn(|i| i as u8 + 1);
        let mut aligned = [0xEE_u8; 32];
        to_aligned(&dense, &offsets, &lens, &mut aligned).expect("aligned");
        assert_eq!(&aligned[..12], &dense[..12]);
        assert_eq!(aligned[16], 13);
        assert_eq!(&aligned[24..26], &[14, 15]);
        let mut again = [0_u8; 15];
        to_dense(&aligned, &offsets, &lens, &mut again).expect("dense");
        assert_eq!(again, dense);
    }

    #[test]
    fn wrong_sizes_and_layouts_are_refused() {
        let mut dense = [0_u8; 7];
        assert_eq!(
            to_dense(&[0; 16], &[0, 8], &[4, 4], &mut dense),
            Err(PackError::Length { want: 8, got: 7 })
        );
        let mut aligned = [0_u8; 16];
        assert_eq!(
            to_aligned(&[0; 8], &[0, 8], &[4, 4], &mut aligned[..10]),
            Err(PackError::Layout),
            "a field past the stride"
        );
        assert_eq!(
            to_aligned(&[0; 8], &[0], &[4, 4], &mut aligned),
            Err(PackError::Layout)
        );
    }
}
