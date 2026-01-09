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
import {AttackInfo} from "@/types";
import AttackDetails from "@/components/attack/attack-details";

export default function AttackSelect({attacks} : {attacks: AttackInfo[]}) {
  const { attack, setAttack, dataset } = useData()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [filtered, setFiltered] = useState(attacks)
  const [selected, setSelected] = useState(attack)

  const selectable = dataset !== null

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
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger className={'text-left w-full'} disabled={!selectable}>
          <AttackDetails clickable={selectable} selectable={selectable} />
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
                  <h3 className="font-semibold text-blue-700">{a.display_name}</h3>
                  <p className="text-sm text-zinc-600 mb-2">{a.description}</p>
                  <div className="text-xs text-zinc-500 flex justify-between">
                    <div>
                      <span className="">Type:</span>{" "}
                      <span className="font-semibold">{a?.type || ''}</span>
                    </div>

                    <div>
                      <span className="">Time:</span>{" "}
                      <span className="font-semibold">{a?.time || ''}</span>
                    </div>
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
