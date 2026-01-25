interface RunGraphProps {
  title?: string
  color?: string
  values: number[]
}

export const RunGraph = ({ title, color, values }: RunGraphProps) => {
  const maxValue = Math.max(...values)

  return (
    <div className="flex flex-col gap-3 w-full">
      {title && <h4 className="font-semibold text-sm text-zinc-800">{title}</h4>}

      <div className="flex gap-3 h-40 items-end">
        {values.map((v, i) => {
          const heightPct = (v / maxValue) * 100

          return (
            <div
              key={i}
              className="flex-1 flex flex-col items-center gap-1"
            >
              <span className="text-[10px] text-zinc-700">
                {v.toFixed(3)}
              </span>

              <div className="relative w-full h-28">
                <div
                  className={`absolute bottom-0 w-full rounded-sm ${color || "bg-zinc-800/80"}`}
                  style={{ height: `${heightPct}%` }}
                />
              </div>

              <span className="text-[10px] text-zinc-500">
                Run {i + 1}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
