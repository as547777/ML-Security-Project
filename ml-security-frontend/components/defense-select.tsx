"use client"
import React from "react"
import { useData } from "@/context/DataContext"

const DefenseSelect = () => {
  const { defense, setDefense } = useData()

  return (
    <div>
      <label>Defense</label>
      <select value={defense} onChange={(e) => setDefense(e.target.value)}>
        <option value="Adversarial Training">Adversarial Training</option>
        <option value="Defensive Distillation">Defensive Distillation</option>
        <option value="Randomization">Randomization</option>
      </select>
    </div>
  )
}

export default DefenseSelect
