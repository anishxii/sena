import React from "react";
import { cn } from "../../lib/cn";

export function Button({ children, className, ...props }) {
  return (
    <button className={cn("ui-button", className)} {...props}>
      {children}
    </button>
  );
}
