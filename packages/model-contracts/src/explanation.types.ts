export interface ShapFeatureContribution {
  featureName: string;            // e.g. "beta_band_power_ch3", "delta_variance_ch1"
  value: number;                  // signed SHAP value; sign drives bar direction/color
}

export interface ShapExplanation {
  baseValue: number;
  features: ShapFeatureContribution[];   // sorted by |value| desc, top-N for display
}
