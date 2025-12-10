

export interface DatasetInfo {
  name: string
  description: string
  type: string
  trainCount: number
  testCount: number
}

export interface AttackInfo {
  name: string
  description: string
  type: string
  params: AttackParams
}

export interface ParamField {
  label: string;
  tooltip: string;
  type: "number" | "string" | "select";
  step?: number;
  options?: string[];
  value: number | string;
}

export type AttackParams = Record<string, ParamField>;

// TODO - napravit isto to samo za DefenseInfo
