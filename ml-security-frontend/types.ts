export interface DatasetInfo {
  name: string
  description: string
  type: string
  trainCount: number
  testCount: number
  classes: string[]
}

export interface ParamField {
  label: string;
  tooltip: string;
  type: "number" | "string" | "select" | "select_class";
  step?: number;
  options?: string[];
  value: number | string;
}

export type ParamsType = Record<string, ParamField>;

export interface AttackInfo {
  name: string
  description: string
  type: string
  params: ParamsType
}

export interface DefenseInfo {
  name: string
  description: string
  type: string
  params: ParamsType
}

