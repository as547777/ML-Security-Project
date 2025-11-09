"use client"
import React from "react"
import { useData } from "@/context/DataContext"
import FieldInput from "@/components/field-input";

const HyperparameterInputs = () => {
  const {
    learningRate, setLearningRate,
    epochs, setEpochs,
    momentum, setMomentum
  } = useData()

  return (
    <div className="flex flex-wrap gap-4">
      <FieldInput
        type="number"
        label={"Learning rate"}
        tooltip={"Controls how much to adjust the model’s weights after each update. Smaller values make training slower but more stable."}
        step={0.001}
        value={learningRate} setValue={setLearningRate} />

      <FieldInput
        type="number"
        label="Epochs"
        tooltip="The number of times the model sees the entire training dataset."
        step={1}
        value={epochs} setValue={setEpochs} />

      <FieldInput
        type="number"
        label="Momentum"
        tooltip="Controls how quickly the model adapts to changes in the training data."
        step={0.01}
        value={momentum} setValue={setMomentum} />
    </div>
  )
}

export default HyperparameterInputs
