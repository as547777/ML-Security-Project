"use client"
import {createContext, PropsWithChildren, useContext, useState} from "react"
import {AttackInfo, DatasetInfo} from "@/types";

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

  // TODO - ovo tu je jako privremeno, hardkodirano je samo za badnets.
  //  neka ideja je da za svaki napad imamo neke dinamicke parametre, jer tipa napadi na zvuk nece imat "trigger size" i slicno
  //  bilo bi kul da kad sa beka povlacimo koji napad zelimo, da on vrati nekakav popis svih parametra pa po tome se dinamicki kreiraju
  //  zbog toga treba nastojati napraviti FieldInputs komponentu koliko god dinamicka moze bit
  //  (to podrazumijeva da treba istrazit kako funkcionira generiranje dinamickih stateova u reactu (yikes))
  //  MALI NASTAVAK -> za badnets mozemo cak dodat "trigger position" i onda birat izmedu center, top left, bottom right...
  //  ^^^ za ovo treba prilagoti FieldInputs heh
  sourceLabel: number
  setSourceLabel: (sourceLabel: number) => void
  targetLabel: number
  setTargetLabel: (targetLabel: number) => void
  poisonRate: number
  setPoisonRate: (poisonRate: number) => void
  triggerSize: number
  setTriggerSize: (triggerSize: number) => void

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

  // Poisoning attack params
  const [sourceLabel, setSourceLabel] = useState(1)
  const [targetLabel, setTargetLabel] = useState(7)
  const [poisonRate, setPoisonRate] = useState(0.9)
  const [triggerSize, setTriggerSize] = useState(4)

  // Defense
  const [defense, setDefense] = useState("")

  return (
    <DataContext.Provider
      value={{
        dataset, setDataset,
        learningRate, setLearningRate,
        epochs, setEpochs,
        momentum, setMomentum,
        attack, setAttack,
        sourceLabel, setSourceLabel,
        targetLabel, setTargetLabel,
        poisonRate, setPoisonRate,
        triggerSize, setTriggerSize,
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
