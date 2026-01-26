"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { cn } from "@/lib/utils"
import { motion } from "framer-motion"
import { Home, Database, Shield, BarChart2, Award } from "lucide-react"

const steps = [
  { name: "Home", path: "/", icon: Home },
  { name: "Dataset", path: "/dataset", icon: Database },
  { name: "Attacks & Defenses", path: "/attack-defense", icon: Shield },
  { name: "Overview", path: "/overview", icon: BarChart2 },
  { name: "Results", path: "/results", icon: Award },
]

export function ProgressNavbar() {
  const pathname = usePathname()

  return (
    <motion.nav
      className="sticky top-0 z-40 flex justify-center py-4 bg-white/70 backdrop-blur-xl shadow-md border-b border-white/30"
      initial={{ y: -40, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      <ul className="flex items-center gap-6 md:gap-10 text-sm font-medium text-zinc-700">
        <div className="absolute bottom-0 left-0 h-[3px] bg-blue-500 transition-all duration-700"
        style={{ width: `${(steps.findIndex(s => s.path === pathname) / (steps.length - 1)) * 100}%` }} />
        {steps.map((step, index) => {
          const isActive = pathname === step.path
          const Icon = step.icon
          return (
            <li key={step.path} className="flex items-center gap-2">
              <Link
                href={step.path}
                className={cn(
                  "flex items-center gap-2 transition-all duration-300 hover:text-blue-600 hover:scale-105",
                  isActive
                    ? "text-blue-600 font-semibold drop-shadow-sm"
                    : "text-zinc-600"
                )}
              >
                <Icon size={18} className={cn(isActive ? "text-blue-500" : "text-zinc-400")} />
                <span>{step.name}</span>
              </Link>
              {index < steps.length - 1 && (
                <span className="text-zinc-400">›</span>
              )}
            </li>
          )
        })}
      </ul>
    </motion.nav>
  )
}