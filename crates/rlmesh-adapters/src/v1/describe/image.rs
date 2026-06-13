//! Summarize a resolved image plan.

use super::super::plans::ImagePlan;
use super::super::pyfmt::py_repr;
use super::super::spec::ImageLayout;

pub(super) fn describe_image(plan: &ImagePlan) -> String {
    let mut steps: Vec<String> = Vec::new();
    if plan.src_layout != ImageLayout::Hwc {
        steps.push(format!("{}->hwc", plan.src_layout.as_str()));
    }
    if plan.flip {
        steps.push("flip 180".to_owned());
    }
    if let Some((height, width)) = plan.size {
        steps.push(format!("resize {height}x{width} ({})", plan.resample));
    }
    if plan.normalize {
        steps.push("normalize /255".to_owned());
    }
    steps.push(plan.dtype.clone());
    if plan.dst_layout != ImageLayout::Hwc {
        steps.push(format!("hwc->{}", plan.dst_layout.as_str()));
    }
    if plan.lead_dims > 0 {
        steps.push(format!("+{} lead dims", plan.lead_dims));
    }
    format!(
        "{} <- image {} ({})",
        py_repr(&plan.model_key),
        py_repr(&plan.env_key),
        steps.join(", ")
    )
}
