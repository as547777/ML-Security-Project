// lib/utils.ts
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: never[]) {
  return twMerge(clsx(inputs))
}
