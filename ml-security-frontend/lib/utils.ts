// lib/utils.ts
import { clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: string[]) {
  return twMerge(clsx(inputs))
}

export const mean = (arr: number[]) =>
  arr.reduce((a, b) => a + b, 0) / arr.length
