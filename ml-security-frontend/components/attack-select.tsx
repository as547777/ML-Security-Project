"use client"

import * as React from "react"
import { useState, useEffect } from "react"
import { useData } from "@/context/DataContext"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"

const attacks = [
  {
    name: "FGSM",
    description:
      "Fast Gradient Sign Method — generates adversarial examples by taking a single step in the direction of the gradient of the loss function.",
    type: "White-box attack",
    complexity: "Low",
  },
  {
    name: "PGD",
    description:
      "Projected Gradient Descent — an iterative variant of FGSM that applies multiple smaller perturbations to maximize adversarial effect.",
    type: "White-box attack",
    complexity: "Medium",
  },
  {
    name: "DeepFool",
    description:
      "An iterative method that finds minimal perturbations to fool classifiers by approximating the decision boundary locally.",
    type: "White-box attack",
    complexity: "High",
  },
]

export default function AttackSelect() {
  const { attack, setAttack } = useData()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [filtered, setFiltered] = useState(attacks)
  const [selected, setSelected] = useState(attack)

  useEffect(() => {
    const handler = setTimeout(() => {
      const term = search.toLowerCase()
      setFiltered(attacks.filter((a) => a.name.toLowerCase().includes(term)))
    }, 200)
    return () => clearTimeout(handler)
  }, [search])

  const handleConfirm = () => {
    setAttack(selected)
    setOpen(false)
  }

  return (
    <div>
      <label className="block text-sm font-medium mb-1 text-zinc-500">Attack</label>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <Button variant="outline" className="w-full justify-baseline text-zinc-700 text-md py-5">
            {attack || "Select an attack"}
          </Button>
        </DialogTrigger>

        <DialogContent className="max-w-lg text-zinc-900">
          <DialogHeader>
            <DialogTitle>Select an Attack</DialogTitle>
          </DialogHeader>

          <input
            placeholder="Search attacks..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3"
          />

          <ScrollArea className="h-116 pr-2">
            <div className="space-y-3">
              {filtered.map((a) => (
                <div
                  key={a.name}
                  onClick={() => setSelected(a.name)}
                  className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                    selected === a.name
                      ? "border-blue-500 bg-blue-50"
                      : "border-zinc-200 hover:bg-zinc-50"
                  }`}
                >
                  <h3 className="font-semibold text-blue-700">{a.name}</h3>
                  <p className="text-sm text-zinc-600 mb-2">{a.description}</p>
                  <div className="text-xs text-zinc-500 flex justify-between">
                    <span>{a.type}</span>
                    <span>Complexity: {a.complexity}</span>
                  </div>
                </div>
              ))}

              {filtered.length === 0 && (
                <p className="text-sm text-zinc-500 text-center py-6">
                  No attacks found.
                </p>
              )}
            </div>
          </ScrollArea>

          <DialogFooter className="mt-4 flex justify-between">
            <Button variant="ghost" onClick={() => setOpen(false)}>
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={!selected}>
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
