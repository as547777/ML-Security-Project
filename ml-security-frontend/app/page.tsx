import { Button } from "@/components/ui/button"
import Link from "next/link"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-zinc-50">
      <h1 className="text-4xl font-bold mb-6 text-zinc-800">
        ML Security Dashboard
      </h1>

      <p className="text-zinc-600 mb-8 text-center max-w-lg">
        Dobrodošli u naš ML Security Project!
      </p>
      <Link href="/dataset">
        <Button>Pokreni sustav</Button>
      </Link>
    </main>
  )
}