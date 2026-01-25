export interface ParamField {
  label: string;
  tooltip: string;
  type: "number" | "string" | "select" | "select_class";
  step?: number;
  min?: number;
  max?: number;
  options?: string[];
  value: number | string;
}

export type ParamsType = Record<string, ParamField>;

export interface DatasetInfo {
  name: string
  display_name: string
  description: string
  type: string
  trainCount: number
  testCount: number
  classes: string[]
}

export interface ModelInfo {
  name: string
  description: string
  category: string
  use_case: string
  models: string[]
}

export interface AttackInfo {
  name: string
  display_name: string
  description: string
  type: string
  time: string
  params: ParamsType
}

export interface DefenseInfo {
  name: string
  display_name: string
  description: string
  type: string
  time: string
  params: ParamsType
}

export interface ResultInfo {
  metrics: {
    attack: {
      patch_area_ratio: number
      poison_rate: number
    }
    deviations: {
      asr: {
        before: {
          mean: number
          std: number
        }
        after: {
          mean: number
          std: number
        }
      }
      clean_acc: {
        before: {
          mean: number
          std: number
        }
        after: {
          mean: number
          std: number
        }
      }
    }
    improvement: {
      acc_drop: number[]
      asr_reduction: number[]
    }
  }
  visualizations: {
    source_image: string
    poisoned_image: string
    residual_image: string
    source_label: number
    prediction_clean: number
    prediction_poisoned: number
    target_label: number
  }[]
}

export interface ResultInfoOld {
  attack_phase: {
    accuracy: number
    asr: number
  }
  defense_phase: {
    accuracy: number
    asr: number
  }
  improvement: {
    asr_reduction: number
    acc_drop: number
  }
  visualizations: {
    source_image: string
    poisoned_image: string
    residual_image: string
    source_label: number
    prediction_clean: number
    prediction_poisoned: number
    target_label: number
  }[]
}