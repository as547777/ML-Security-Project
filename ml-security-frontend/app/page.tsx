"use client"

import { motion } from "framer-motion"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Sparkles } from "lucide-react"

export default function HomePage() {
  return (
    <motion.main
      className="flex flex-col items-center justify-center min-h-[calc(100vh-100px)] px-6 text-center"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <div className="bg-white/10 backdrop-blur-lg border border-white/20 shadow-2xl rounded-3xl p-12 w-full max-w-4xl">
        <div className="flex flex-col items-center gap-6">
          <motion.h1
            className="header-text text-5xl md:text-6xl p-2"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1, duration: 0.8 }}
          >
            BackdoorHub
          </motion.h1>

          <motion.p
            className="text-lg md:text-xl text-zinc-300 max-w-2xl leading-relaxed"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2, duration: 0.8 }}
          >
          </motion.p>

          <motion.div
            className="pt-4"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3, duration: 0.6 }}
          >
            <Link href="/dataset">
              <Button
                size="lg"
                className="px-8 py-6 text-lg font-semibold bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_15px_rgba(59,130,246,0.6)] hover:shadow-[0_0_30px_rgba(59,130,246,0.9)] transition-all duration-300"
              >
                <Sparkles className="mr-2 h-5 w-5" />
                Setup Environment
              </Button>
            </Link>
          </motion.div>
        </div>
      </div>

      <motion.p
        className="mt-10 text-sm text-zinc-400"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.8 }}
      >
        Project: <span className="text-blue-400">ML Security Framework</span>
      </motion.p>
    </motion.main>
  )
}