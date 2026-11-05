import { describe, expect, it } from "vitest";
import { isStepKey } from "./DateScrubber";

function pressOn(element: HTMLElement, init: KeyboardEventInit): boolean {
  let result = false;
  element.addEventListener("keydown", (event) => {
    result = isStepKey(event);
  });
  document.body.append(element);
  element.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, ...init }));
  element.remove();
  return result;
}

describe("isStepKey", () => {
  it("steps on plain left and right arrows", () => {
    expect(pressOn(document.createElement("div"), { key: "ArrowLeft" })).toBe(true);
    expect(pressOn(document.createElement("div"), { key: "ArrowRight" })).toBe(true);
  });

  it("ignores other keys and arrows with a modifier", () => {
    expect(pressOn(document.createElement("div"), { key: "ArrowUp" })).toBe(false);
    expect(pressOn(document.createElement("div"), { key: "ArrowLeft", altKey: true })).toBe(false);
  });

  it("leaves arrows alone in form controls and scrollable tables", () => {
    expect(pressOn(document.createElement("input"), { key: "ArrowLeft" })).toBe(false);
    expect(pressOn(document.createElement("select"), { key: "ArrowLeft" })).toBe(false);
    const region = document.createElement("div");
    region.setAttribute("role", "region");
    expect(pressOn(region, { key: "ArrowRight" })).toBe(false);
  });
});
