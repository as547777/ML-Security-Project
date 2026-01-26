"use client"
import React from "react"
import {
  ImageIcon,
  Table2,
  Music,
  Video,
  FileQuestion,
} from "lucide-react"

interface Props {
  type: string | undefined
  size?: number
  color?: string
}

const DatasetTypeIcon = ({ type, size = 5, color }: Props) => {
  const iconSize = `w-${size} h-${size}`

  const withColor = (defaultColor: string) =>
    `${iconSize} ${color ?? defaultColor}`

  const getIcon = () => {
    switch (type?.toLowerCase()) {
      case "image":
      case "image (grayscale)":
      case "image (rgb)":
        return <ImageIcon className={withColor("text-blue-600")} />

      case "text":
      case "tabular":
        return <Table2 className={withColor("text-emerald-600")} />

      case "audio":
        return <Music className={withColor("text-purple-600")} />

      case "video":
        return <Video className={withColor("text-rose-600")} />

      default:
        return <FileQuestion className={withColor("text-zinc-500")} />
    }
  }

  return <div className="flex items-center justify-center">{getIcon()}</div>
}

export default DatasetTypeIcon
