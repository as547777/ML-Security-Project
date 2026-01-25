import React from 'react';
import Image from "next/image";
import {ResultInfo} from "@/types";
import ParamCard from "@/components/param-card";

const PoisonExample = ({examples, limit = 1}: {examples : ResultInfo['visualizations'], limit?: number}) => {
  return (
    examples.length !== 0 && examples && (
      <div className={'grid grid-cols-1 lg:grid-cols-3 gap-2'}>
        {examples.slice(0, limit).map((example, index) => (
          <ParamCard
            className={'col-span-1'}
            key={index}
            label={`Example ${index + 1}`}
            value={
              <div className={'flex items-center justify-between px-4 2xl:px-16'}>
                <div>
                  <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Original</div>
                  <Image
                    src={`data:image/png;base64,${example.source_image}`}
                    alt={"Original Image"}
                    width={100}
                    height={100}
                  />
                </div>
                <span className={'text-2xl mx-1'}>+</span>
                <div>
                  <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Residual</div>
                  <Image
                    src={`data:image/png;base64,${example.residual_image}`}
                    alt={"Residual Image"}
                    width={100}
                    height={100}
                  />
                </div>
                <span className={'text-2xl mx-1'}>→</span>
                <div>
                  <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Poisoned</div>
                  <Image
                    src={`data:image/png;base64,${example.poisoned_image}`}
                    alt={"Poisoned Image"}
                    width={100}
                    height={100}
                  />
                </div>
              </div>
            }
          />
        ))}
      </div>
    )
  );
};

export default PoisonExample;