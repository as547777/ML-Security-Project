"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Button } from "@/components/ui/button"

export default function VisualizationPage() {
  return (
    <motion.main
      className="flex flex-col items-center justify-center min-h-[calc(100vh-100px)] p-8"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <h1 className="text-3xl font-bold text-zinc-800 mb-6">
        Vizualizacija i grafovi
      </h1>

      <div className="bg-white/80 backdrop-blur-md shadow-lg rounded-3xl p-10 w-full max-w-2xl border border-white/30">
        <p className="text-zinc-600 mb-4">
          Ovdje će se prikazivati grafovi treninga, gubitka i točnosti modela.
        </p>

        <div className="border border-dashed border-zinc-300 rounded-lg h-64 flex items-center justify-center text-zinc-400">
          (Ovdje će ići grafovi)
        </div>

        <Link href="/results">
          <Button className="mt-8 w-full">Prikaži rezultate</Button>
        </Link>
      </div>
    </motion.main>
  )
}