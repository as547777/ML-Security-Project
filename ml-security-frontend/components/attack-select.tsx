"use client"
import React from "react"
import { useData } from "@/context/DataContext"

const AttackSelect = () => {
  const { attack, setAttack } = useData()

  return (
    <div>
      <label>Attack</label>
      <select value={attack} onChange={(e) => setAttack(e.target.value)}>
        <option value="FGSM">FGSM</option>
        <option value="PGD">PGD</option>
        <option value="DeepFool">DeepFool</option>
      </select>
    </div>
  )
}

export default AttackSelect
