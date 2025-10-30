"use client"
import React from "react"
import {
  ImageIcon,
  Table2,
  Music,
  Video,
  FileQuestion,
} from "lucide-react"

const DatasetTypeIcon = ({ type }: { type: string | undefined }) => {
  const getIcon = () => {
    switch (type?.toLowerCase()) {
      case "image":
      case "image (grayscale)":
      case "image (rgb)":
        return <ImageIcon className="w-5 h-5 text-blue-600" />

      case "text":
      case "tabular":
        return <Table2 className="w-5 h-5 text-emerald-600" />

      case "audio":
        return <Music className="w-5 h-5 text-purple-600" />

      case "video":
        return <Video className="w-5 h-5 text-rose-600" />

      default:
        return <FileQuestion className="w-5 h-5 text-zinc-500" />
    }
  }

  return <div className="flex items-center justify-center">{getIcon()}</div>
}

export default DatasetTypeIcon
