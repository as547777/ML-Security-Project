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
import {DefenseInfo} from "@/types";

export default function DefenseSelect({defenses}: {defenses: DefenseInfo[]}) {
  const { defense, setDefense } = useData()
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState("")
  const [filtered, setFiltered] = useState(defenses)
  const [selected, setSelected] = useState(defense)

  useEffect(() => {
    const handler = setTimeout(() => {
      const term = search.toLowerCase()
      setFiltered(defenses.filter((d) => d.name.toLowerCase().includes(term)))
    }, 200)
    return () => clearTimeout(handler)
  }, [search])

  const handleConfirm = () => {
    setDefense(selected)
    setOpen(false)
  }

  return (
    <div>
      <label className="block text-sm font-medium mb-1 text-zinc-500">Defense</label>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogTrigger asChild>
          <div className="select-div">
            {defense?.name ? (
              <div>
                <h1 className={'text-base mb-1'}>{defense.name}</h1>
                <p className="text-sm text-zinc-500 mb-2">{defense.description}</p>
                <div className="text-xs text-zinc-400 flex justify-between">
                  <span>ovdje dodati nesto</span>
                  <span>i tu</span>
                </div>
              </div>
            ) : (
              <span className={'text-base'}>Select a defense</span>
            )}
          </div>
        </DialogTrigger>

        <DialogContent className="max-w-lg text-zinc-900">
          <DialogHeader>
            <DialogTitle>Select a Defense</DialogTitle>
          </DialogHeader>

          <input
            placeholder="Search defenses..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="mb-3"
          />

          <ScrollArea className="h-116 pr-2">
            <div className="space-y-3">
              {filtered.map((d) => (
                <div
                  key={d.name}
                  onClick={() => setSelected(d)}
                  className={`cursor-pointer rounded-lg border p-3 transition-colors ${
                    selected === d
                      ? "border-blue-500 bg-blue-50"
                      : "border-zinc-200 hover:bg-zinc-50"
                  }`}
                >
                  <h3 className="font-semibold text-blue-700">{d.name}</h3>
                  <p className="text-sm text-zinc-600 mb-2">{d.description}</p>
                  <div className="text-xs text-zinc-500 flex justify-between">
                    <span>{d.type}</span>
                    <span>teski mrnjau</span>
                  </div>
                </div>
              ))}

              {filtered.length === 0 && (
                <p className="text-sm text-zinc-500 text-center py-6">
                  No defenses found.
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
