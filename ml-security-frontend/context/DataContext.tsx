"use client"
import {createContext, PropsWithChildren, useContext, useState} from "react"
import {AttackInfo, ParamsType, DatasetInfo, DefenseInfo, ModelInfo} from "@/types";

interface DataType {
  dataset: DatasetInfo | null
  setDataset: (dataset: DatasetInfo | null) => void

  modelFamily: ModelInfo | null
  setModelFamily: (modelFamily: ModelInfo | null) => void
  model: string
  setModel: (model: string) => void
  learningRate: number
  setLearningRate: (learningRate: number) => void
  epochs: number
  setEpochs: (epochs: number) => void
  momentum: number
  setMomentum: (momentum: number) => void
  batchSize: number
  setBatchSize: (batchSize: number) => void
  lossFunction: string
  setLossFunction: (lossFunction: string) => void
  optimizer: string
  setOptimizer: (optimizer: string) => void

  attack: AttackInfo | null
  setAttack: (attack: AttackInfo | null) => void
  attackParams: ParamsType | undefined
  updateAttackParams: (key: keyof ParamsType, value: number | string) => void

  defense: DefenseInfo | null
  setDefense: (defense: DefenseInfo | null) => void
  defenseParams: ParamsType | undefined
  updateDefenseParams: (key: keyof ParamsType, value: number | string) => void
}

const DataContext = createContext<DataType | null>(null)

export function DataProvider({ children }: PropsWithChildren) {
  // Dataset
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)

  // Hyperparameters
  const [modelFamily, setModelFamily] = useState<ModelInfo | null>(null)
  const [model, setModel] = useState<string>('')
  const [learningRate, setLearningRate] = useState(0.01)
  const [epochs, setEpochs] = useState(10)
  const [momentum, setMomentum] = useState(0.9)
  const [batchSize, setBatchSize] = useState(32)

  const [lossFunction, setLossFunction] = useState("CrossEntropyLoss")
  const [optimizer, setOptimizer] = useState("SGD")
  // TODO - picek veli da treba dodat izgled neuronske mreze

  // Attack
  const [attack, setAttackInfo] = useState<AttackInfo | null>(null)
  const [attackParams, setAttackParams] = useState<ParamsType | undefined>(attack?.params);
  const updateAttackParams = (key: keyof ParamsType, value: number | string) => {
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
  const setAttack = (attack: AttackInfo | null) => {
    setAttackInfo(attack);
    setAttackParams(attack?.params)
  }

  // Defense
  const [defense, setDefenseInfo] = useState<DefenseInfo | null>(null)
  const [defenseParams, setDefenseParams] = useState<ParamsType | undefined>(defense?.params);
  const updateDefenseParams = (key: keyof ParamsType, value: number | string) => {
    setDefenseParams(prev => {
      if (!prev) return prev;
      return {
        ...prev,
        [key]: {
          ...prev[key],
          value,
        },
      };
    });
  }
  const setDefense = (defense: DefenseInfo | null) => {
    setDefenseInfo(defense);
    setDefenseParams(defense?.params)
  }

  return (
    <DataContext.Provider
      value={{
        dataset, setDataset,
        modelFamily, setModelFamily,
        model, setModel,
        learningRate, setLearningRate,
        epochs, setEpochs,
        momentum, setMomentum,
        batchSize, setBatchSize,
        lossFunction, setLossFunction,
        optimizer, setOptimizer,
        attack, setAttack,
        attackParams, updateAttackParams,
        defense, setDefense,
        defenseParams, updateDefenseParams
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