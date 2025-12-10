"use client"
import {createContext, PropsWithChildren, useContext, useEffect, useState} from "react"
import {AttackInfo, AttackParams, DatasetInfo} from "@/types";

interface DataType {
  dataset: DatasetInfo | null
  setDataset: (dataset: DatasetInfo | null) => void

  learningRate: number
  setLearningRate: (learningRate: number) => void
  epochs: number
  setEpochs: (epochs: number) => void
  momentum: number
  setMomentum: (momentum: number) => void

  attack: AttackInfo | null
  setAttack: (attack: AttackInfo | null) => void
  attackParams: AttackParams | undefined
  updateAttackParams: (key: keyof AttackParams, value: number | string) => void

  defense: string
  setDefense: (defense: string) => void
}

const DataContext = createContext<DataType | null>(null)

export function DataProvider({ children }: PropsWithChildren) {
  // Dataset
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)

  // Hyperparameters
  const [learningRate, setLearningRate] = useState(0.01)
  const [epochs, setEpochs] = useState(10)
  const [momentum, setMomentum] = useState(0.9)
  // mozda dodat batch size

  // Attack
  const [attack, setAttack] = useState<AttackInfo | null>(null)
  const [attackParams, setAttackParams] = useState<AttackParams | undefined>(attack?.params);
  const updateAttackParams = (key: keyof AttackParams, value: number | string) => {
    setAttackParams(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: {
          ...prev[key],
          value,
        },
      };
    });
  };

  // Defense
  const [defense, setDefense] = useState("")

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setAttackParams(attack?.params);
  }, [attack]);

  return (
    <DataContext.Provider
      value={{
        dataset, setDataset,
        learningRate, setLearningRate,
        epochs, setEpochs,
        momentum, setMomentum,
        attack, setAttack,
        attackParams, updateAttackParams,
        defense, setDefense,
      }}
    >
      {children}
    </DataContext.Provider>
  )
}

export const useData = () => {
  const context = useContext(DataContext)
  if (!context) {
    throw new Error("useData must be used within a DataProvider")
  }
  return context
}

// const params: AttackParams = {
//   source_label: {
//     label: "Source label",
//     tooltip: "Label of the class that will be poisoned (e.g., 1)",
//     type: "number",
//     step: 1,
//     value: 1
//   },
//   target_label: {
//     label: "Target label",
//     tooltip: "Label of the class that poisoned samples should be misclassified as (e.g., 7)",
//     type: "number",
//     step: 1,
//     value: 7
//   },
//   poison_rate: {
//     label: "Poison rate",
//     tooltip: "Fraction of samples from the source class to poison (0–1)",
//     type: "number",
//     step: 0.01,
//     value: 0.2
//   },
//   trigger_size: {
//     label: "Trigger size",
//     tooltip: "Size of the injected trigger patch (e.g., 4 for a 4×4 pixel square)",
//     type: "number",
//     step: 1,
//     value: 4
//   }
// }