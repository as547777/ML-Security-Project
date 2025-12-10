"use client"
import React from "react"
import { useData } from "@/context/DataContext"
import FieldInput from "@/components/field-input";

interface Props {
  optimizers: {name: string, description: string}[],
  lossFunctions: {name: string, description: string}[]
}

const HyperparameterInputs = ({optimizers, lossFunctions} : Props) => {
  const {
    learningRate, setLearningRate,
    epochs, setEpochs,
    momentum, setMomentum,
    optimizer, setOptimizer,
    lossFunction, setLossFunction
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
      
      <FieldInput
        label={"Loss function"}
        tooltip={"Chooses how the model values loss per example"}
        type={'select'}
        options={lossFunctions.map(loss => loss.name)}
        value={lossFunction} setValue={setLossFunction} />

      <FieldInput
        label={"Optimizer"}
        tooltip={"Chooses how the model handles weight updates while training."}
        type={"select"}
        options={optimizers.map(opt => opt.name)}
        value={optimizer} setValue={setOptimizer} />
    </div>
  )
}

export default HyperparameterInputs
