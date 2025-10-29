"use client"
import React from "react"
import { useData } from "@/context/DataContext"

const HyperparameterInputs = () => {
  const { learningRate, setLearningRate, epochs, setEpochs } = useData()

  return (
    <>
      <div>
        <label>Learning Rate</label>
        <input
          type="number"
          step="0.001"
          value={learningRate}
          onChange={(e) => setLearningRate(parseFloat(e.target.value))}
        />
      </div>

      <div>
        <label>Broj epoha</label>
        <input
          type="number"
          value={epochs}
          onChange={(e) => setEpochs(parseInt(e.target.value))}
        />
      </div>
    </>
  )
}

export default HyperparameterInputs
