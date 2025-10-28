import "./globals.css"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { ProgressNavbar } from "@/components/ui/progress-navbar"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "ML Security Dashboard",
  description: "Vizualni sustav za sigurnost neuronskih mreža",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} relative min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800 text-white overflow-x-hidden`}>
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="absolute -top-40 -left-40 w-[600px] h-[600px] rounded-full bg-blue-500/30 blur-[160px] animate-pulse"></div>
          <div className="absolute bottom-0 right-0 w-[700px] h-[700px] rounded-full bg-purple-500/20 blur-[180px] animate-[spin_40s_linear_infinite]"></div>
        </div>

        <ProgressNavbar />
        <main className="relative z-10">{children}</main>
      </body>
    </html>
  )
}
