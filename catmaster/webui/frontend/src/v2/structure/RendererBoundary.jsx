import { Component } from "react";

export default class RendererBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    window.__catmasterLastRendererError = {
      message: error?.message || String(error),
      stack: error?.stack || "",
      componentStack: info?.componentStack || "",
    };
    console.error("CatMaster renderer boundary", error);
    this.props.onError?.(error?.message || String(error));
  }

  render() {
    if (this.state.error) {
      return (
        <div className="v2-renderer-fallback" role="alert">
          <strong>The primary 3D editor could not start.</strong>
          <p>{this.state.error.message || String(this.state.error)}</p>
          {this.props.fallback}
        </div>
      );
    }
    return this.props.children;
  }
}
