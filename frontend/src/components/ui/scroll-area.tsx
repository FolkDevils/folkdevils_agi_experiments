import { ReactNode } from 'react';
import { clsx } from 'clsx';

interface ScrollAreaProps {
  children: ReactNode;
  className?: string;
}

export function ScrollArea({ children, className }: ScrollAreaProps) {
  return (
    <div className={clsx('overflow-auto', className)}>
      {children}
    </div>
  );
} 