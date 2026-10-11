import React from 'react';
import { Activity, Cpu, CheckCircle2, AlertTriangle, ShieldAlert, RefreshCw } from 'lucide-react';
import { ModelStatusResponse } from '../types';

interface HeaderProps {
  activeTab: 'screening' | 'status' | 'evaluation';
  setActiveTab: (tab: 'screening' | 'status' | 'evaluation') => void;
  modelStatus: ModelStatusResponse | null;
  isBackendConnected: boolean;
  isLoadingStatus: boolean;
  onRefreshStatus: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  modelStatus,
  isBackendConnected,
  isLoadingStatus,
  onRefreshStatus,
}) => {
  return (
    <header className="border-b border-slate-200 bg-white shadow-xs">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-4 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <div className="flex items-center gap-2.5">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-cyan-700 text-white shadow-xs">
                <Activity className="h-5 w-5" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight text-slate-900">
                  Detecting Abnormal Bone Fracture Using SPECT Images
                </h1>
                <p className="text-xs text-slate-500">
                  Planar & Tomographic Bone Scintigraphy • PyTorch ResNet-18 • Grad-CAM Spatial Localization
                </p>
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {/* Research Use Badge */}
            <span className="inline-flex items-center gap-1 rounded-md bg-amber-50 px-2.5 py-1 text-xs font-medium text-amber-800 ring-1 ring-amber-200">
              <ShieldAlert className="h-3.5 w-3.5" />
              Research Use Only
            </span>

            {/* Backend Connectivity Status */}
            <span
              className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ring-1 ${
                isBackendConnected
                  ? 'bg-emerald-50 text-emerald-800 ring-emerald-200'
                  : 'bg-rose-50 text-rose-800 ring-rose-200'
              }`}
            >
              <span
                className={`h-2 w-2 rounded-full ${
                  isBackendConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
                }`}
              />
              {isBackendConnected ? 'FastAPI Connected' : 'FastAPI Offline'}
            </span>

            {/* Checkpoint Status */}
            {isBackendConnected && (
              <span
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium ring-1 ${
                  modelStatus?.checkpoint_loaded
                    ? 'bg-blue-50 text-blue-800 ring-blue-200'
                    : 'bg-amber-50 text-amber-800 ring-amber-200'
                }`}
              >
                {modelStatus?.checkpoint_loaded ? (
                  <>
                    <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />
                    Checkpoint Loaded
                  </>
                ) : (
                  <>
                    <AlertTriangle className="h-3.5 w-3.5 text-amber-600" />
                    No Checkpoint
                  </>
                )}
              </span>
            )}

            {/* Hardware Device */}
            {isBackendConnected && modelStatus && (
              <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-1 text-xs text-slate-700">
                <Cpu className="h-3.5 w-3.5 text-slate-500" />
                {modelStatus.device.toUpperCase()}
              </span>
            )}

            {/* Refresh Button */}
            <button
              onClick={onRefreshStatus}
              disabled={isLoadingStatus}
              title="Refresh connection and checkpoint status"
              className="inline-flex items-center justify-center rounded-md border border-slate-300 bg-white p-1.5 text-slate-600 hover:bg-slate-50 disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${isLoadingStatus ? 'animate-spin' : ''}`} />
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex space-x-1 border-t border-slate-100 pt-2">
          <button
            onClick={() => setActiveTab('screening')}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'screening'
                ? 'border-cyan-600 text-cyan-700'
                : 'border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-800'
            }`}
          >
            Clinical Screening & Grad-CAM
          </button>
          <button
            onClick={() => setActiveTab('status')}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'status'
                ? 'border-cyan-600 text-cyan-700'
                : 'border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-800'
            }`}
          >
            Model & Weights Status
          </button>
          <button
            onClick={() => setActiveTab('evaluation')}
            className={`border-b-2 px-4 py-2 text-sm font-medium transition-colors ${
              activeTab === 'evaluation'
                ? 'border-cyan-600 text-cyan-700'
                : 'border-transparent text-slate-600 hover:border-slate-300 hover:text-slate-800'
            }`}
          >
            Dataset & Held-Out Evaluation
          </button>
        </div>
      </div>
    </header>
  );
};
