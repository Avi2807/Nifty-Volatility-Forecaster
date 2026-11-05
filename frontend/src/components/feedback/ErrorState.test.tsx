import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { API_URL, ApiError } from "../../api/client";
import { DataSourceBanner } from "../layout/DataSourceBanner";
import { ErrorState } from "./ErrorState";

afterEach(cleanup);

describe("ErrorState", () => {
  it("shows the backend URL and start command when the backend can't be reached", () => {
    render(<ErrorState error={new ApiError("Can't reach the backend.")} />);
    expect(screen.getByRole("alert").textContent).toContain(API_URL);
    expect(screen.getByRole("alert").textContent).toContain("uvicorn src.api.main:app");
  });

  it("shows only the backend's message for a refused request", () => {
    render(<ErrorState error={new ApiError("Not a test date.", 404)} />);
    expect(screen.getByRole("alert").textContent).toContain("Not a test date.");
    expect(screen.getByRole("alert").textContent).not.toContain("uvicorn");
  });
});

describe("DataSourceBanner", () => {
  it("adds nothing for real results", () => {
    const { container } = render(<DataSourceBanner source="real" />);
    expect(container.innerHTML).toBe("");
  });

  it("labels mock and synthetic data", () => {
    render(<DataSourceBanner source="mock" />);
    expect(screen.getByRole("status").textContent).toContain("Mock mode");
    cleanup();
    render(<DataSourceBanner source="synthetic" />);
    expect(screen.getByRole("status").textContent).toContain("must not be reported");
  });
});
