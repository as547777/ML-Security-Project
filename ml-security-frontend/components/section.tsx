import React, {PropsWithChildren} from 'react';

interface Props {
  title: string;
}

const Section = ({ title, children }: PropsWithChildren<Props>) => (
  <div className="rounded-xl bg-white backdrop-blur-sm border border-zinc-200/50 overflow-hidden">
    <div className="px-4 pt-3 pb-2 flex items-center gap-2">
      <h3 className="text-sm font-medium text-zinc-500">{title}</h3>
    </div>

    <div className="px-4 pb-4 pt-2">
      {children}
    </div>
  </div>
)
export default Section;