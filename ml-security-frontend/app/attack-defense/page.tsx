"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function AttackDefensePage() {
  return (
    <motion.main
      className="flex flex-col items-center justify-center min-h-[calc(100vh-100px)] p-8"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <h1 className="text-3xl font-bold text-zinc-800 mb-6">
        Odabir napada i obrana
      </h1>

      <div className="bg-white/80 backdrop-blur-md shadow-lg rounded-3xl p-10 w-full max-w-2xl border border-white/30">
        <form className="flex flex-col gap-4">
          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Napad
            </label>
            <select className="w-full border border-zinc-300 rounded-md p-2 focus:ring-2 focus:ring-zinc-400">
              <option>FGSM</option>
              <option>PGD</option>
              <option>DeepFool</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-zinc-700 mb-1">
              Obrana
            </label>
            <select className="w-full border border-zinc-300 rounded-md p-2 focus:ring-2 focus:ring-zinc-400">
              <option>Adversarial Training</option>
              <option>Defensive Distillation</option>
              <option>Randomization</option>
            </select>
          </div>

          <Link href="/visualization">
            <Button className="mt-6 w-full">Nastavi</Button>
          </Link>
        </form>
      </div>
    </motion.main>
  )
}