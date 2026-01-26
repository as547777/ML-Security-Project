import React from 'react';
import {Tooltip, TooltipContent, TooltipProvider, TooltipTrigger} from "@/components/ui/tooltip";
import {Info} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";

interface FieldInputProps<T extends string | number> {
  label: string;
  tooltip: string;
  type: 'number' | 'string' | 'select' | 'select_class';
  step?: number;
  min?: number;
  max?: number;
  options?: string[];
  value: T;
  setValue: ((value: T) => void);
}

const FieldInput = <T extends string | number>(params : FieldInputProps<T>) => {
  const field = () => {
    if (params.type === "number" || params.type === "string") {
      return (
        <input
          type={params.type}
          step={params.step}
          min={params.min}
          max={params.max}
          value={params.value}
          onChange={(e) => {
            const value =
              params.type === "number"
                ? (parseFloat(e.target.value) as T) : (e.target.value as T);
            params.setValue(value);
          }}
          className="param-input"
        />
      )
    } else if (params.type === "select") {
      const stringValue = (params.value).toString();

      return (
        <Select
          value={stringValue}
          onValueChange={(e) => {
            const value =
              params.type === "number"
                ? (parseFloat(e) as T) : (e as T);
            params.setValue(value);
          }}
        >
          <SelectTrigger className="param-input">
            <SelectValue>
              {stringValue}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {params.options?.map((option) => (
                <SelectItem key={option} value={option.toString()}>
                  {option}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      );
    }
  }

  return (
    <TooltipProvider delayDuration={150}>
      <div className="param-container">
        <div className="flex w-full justify-between gap-1 mb-1">
          <label className="param-label">{params.label}</label>
          <Tooltip>
            <TooltipTrigger asChild>
              <Info
                size={14}
                className="text-zinc-400 hover:text-blue-500 transition-colors"
              />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs text-sm text-zinc-200 pb-2">
              {params.tooltip}
            </TooltipContent>
          </Tooltip>
        </div>

        {field()}
      </div>
    </TooltipProvider>
  );
};

export default FieldInput;