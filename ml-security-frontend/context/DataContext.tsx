"use client"
import {createContext, PropsWithChildren, useContext, useState} from "react"

interface DataType {
  dataset: string
  setDataset: (dataset: string) => void

  learningRate: number
  setLearningRate: (learningRate: number) => void

  epochs: number
  setEpochs: (epochs: number) => void

  attack: string
  setAttack: (attack: string) => void

  defense: string
  setDefense: (defense: string) => void
}

const DataContext = createContext<DataType | null>(null)

export function DataProvider({ children }: PropsWithChildren) {
  // Dataset
  const [dataset, setDataset] = useState("")

  // Hyperparameters
  const [learningRate, setLearningRate] = useState(0.01)
  const [epochs, setEpochs] = useState(10)

  // Attack
  const [attack, setAttack] = useState("")

  // Defense
  const [defense, setDefense] = useState("")

  return (
    <DataContext.Provider
      value={{
        dataset, setDataset,
        learningRate, setLearningRate,
        epochs, setEpochs,
        attack, setAttack,
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
