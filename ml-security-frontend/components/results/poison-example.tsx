import React from 'react';
import Image from "next/image";
import Section from "@/components/section";
import {ResultInfo} from "@/types";

const PoisonExample = ({examples}: {examples : ResultInfo['visualizations']}) => {
  return (
    <Section title={"Poison visualization"}>
      <div className={'flex items-center justify-between mb-2 px-15 lg:px-10'}>
        <div>
          <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Original</div>
          <Image
            src={`data:image/png;base64,${examples[0].source_image}`}
            alt={"Original Image"}
            width={100}
            height={100}
          />
        </div>
        <span className={'text-2xl mx-1'}>+</span>
        <div>
          <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Residual</div>
          <Image
            src={`data:image/png;base64,${examples[0].residual_image}`}
            alt={"Residual Image"}
            width={100}
            height={100}
          />
        </div>
        <span className={'text-2xl mx-1'}>→</span>
        <div>
          <div className={'text-xs font-medium text-zinc-500 text-center mb-1'}>Poisoned</div>
          <Image
            src={`data:image/png;base64,${examples[0].poisoned_image}`}
            alt={"Poisoned Image"}
            width={100}
            height={100}
          />
        </div>
      </div>
    </Section>
  );
};

export default PoisonExample;