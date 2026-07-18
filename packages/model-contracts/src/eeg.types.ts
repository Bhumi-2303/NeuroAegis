export interface GraphDataPoint {
  timestamp: string;
  value: number;
  channel?: string;
}

export interface FrequencyBandData {
  gamma: GraphDataPoint[];
  beta: GraphDataPoint[];
  alpha: GraphDataPoint[];
  theta: GraphDataPoint[];
  delta: GraphDataPoint[];
}
