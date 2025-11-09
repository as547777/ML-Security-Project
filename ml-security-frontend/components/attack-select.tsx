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
    name: "BadNets",
    description:
      "Poisoning the dataset by injecting examples with malicious modifications (triggers) into the training data, causing the model to misclassify them when the trigger is present.",
    type: "White-box attack",
    complexity: "Low",
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
          <div className="select-div">
            {attack?.name ? (
              <div>
                <h1 className={'text-base mb-1'}>{attack.name}</h1>
                <p className="text-sm text-zinc-500 mb-2">{attack.description}</p>
                <div className="text-xs text-zinc-400 flex justify-between">
                  <span>ovdje dodati nesto</span>
                  <span>i tu</span>
                </div>
              </div>
            ) : (
              <span className={'text-base'}>Select an attack</span>
            )}
          </div>
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
                  onClick={() => setSelected(a)}
                  className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                    selected === a
                      ? "border-blue-500 bg-blue-50"
                      : "border-zinc-200 hover:bg-zinc-50"
                  }`}
                >
                  <h3 className="font-semibold text-blue-700">{a.name}</h3>
                  <p className="text-sm text-zinc-600 mb-2">{a.description}</p>
                  <div className="text-xs text-zinc-500 flex justify-between">
                    <span>ovdje dodati nesto</span>
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
