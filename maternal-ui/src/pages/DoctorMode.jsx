import React, { useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Activity, 
  BarChart3, 
  Lightbulb, 
  Filter, 

  Layers,
  TrendingUp
} from 'lucide-react';
import { motion } from 'framer-motion';

import Header from '@/components/maternal/Header';
import InputForm from '@/components/maternal/InputForm';
import ResultsPanel from '@/components/maternal/ResultsPanel';
import ModelComparison from '@/components/maternal/doctor/ModelComparison';
import ExplainabilityViz from '@/components/maternal/doctor/ExplainabilityViz';
import FeatureSelection from '@/components/maternal/doctor/FeatureSelection';

import PredictionsTable from '@/components/maternal/PredictionsTable';
import WhatIfSimulator from '@/components/maternal/doctor/WhatIfSimulator';
import RiskFactorPanel from '@/components/maternal/doctor/RiskFactorPanel';
import ConfidenceGauge from '@/components/maternal/doctor/ConfidenceGauge';
import ClinicalRecommendations from '@/components/maternal/doctor/ClinicalRecommendations';
import PopulationContext from '@/components/maternal/doctor/PopulationContext';
import DoctorAnnotations from '@/components/maternal/doctor/DoctorAnnotations';
import ExportReport from '@/components/maternal/doctor/ExportReport';

// Simulated prediction function - replace with actual API call
const simulatePrediction = (data) => {
  const riskScore = 
    (data.systolicBP > 140 ? 30 : data.systolicBP > 130 ? 15 : 0) +
    (data.diastolicBP > 90 ? 20 : data.diastolicBP > 85 ? 10 : 0) +
    (data.bloodSugar > 140 ? 25 : data.bloodSugar > 120 ? 15 : 0) +
    (data.age > 40 ? 15 : data.age > 35 ? 10 : 0) +
    (data.heartRate > 100 ? 10 : data.heartRate > 90 ? 5 : 0);

  let riskLevel = 'low';
  let confidence = 92;
  let probabilities = { low: 75, mid: 20, high: 5 };
  let score = riskScore + 15;

  if (riskScore > 50) {
    riskLevel = 'high';
    confidence = 87;
    probabilities = { low: 10, mid: 25, high: 65 };
    score = Math.min(100, riskScore + 20);
  } else if (riskScore > 25) {
    riskLevel = 'mid';
    confidence = 82;
    probabilities = { low: 25, mid: 55, high: 20 };
    score = riskScore + 18;
  }

  return { riskLevel, confidence, probabilities, score };
};

export default function DoctorMode() {
  const [formData, setFormData] = useState({
    age: 28,
    systolicBP: 120,
    diastolicBP: 80,
    bloodSugar: 95,
    bodyTemp: 98.6,
    heartRate: 75,
  });
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handlePredict = async () => {
    setIsLoading(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    const prediction = simulatePrediction(formData);
    setResult(prediction);
    setIsLoading(false);
  };

  return (
    <div className="space-y-8">
      <Header 
        title="Doctor Interface" 
        subtitle="Advanced clinical analytics and model insights"
      />

      <Tabs defaultValue="prediction" className="w-full">
        <TabsList className="bg-slate-800/50 border border-slate-700/50 p-1 flex flex-wrap gap-1 h-auto">
          <TabsTrigger value="prediction" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <Activity className="w-4 h-4 mr-1 sm:mr-2" />
            <span className="hidden sm:inline">Prediction</span>
            <span className="sm:hidden">Predict</span>
          </TabsTrigger>
          <TabsTrigger value="analysis" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <TrendingUp className="w-4 h-4 mr-1 sm:mr-2" />
            <span className="hidden sm:inline">Analysis</span>
            <span className="sm:hidden">Analyze</span>
          </TabsTrigger>
          <TabsTrigger value="models" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <Layers className="w-4 h-4 mr-1 sm:mr-2" />
            Models
          </TabsTrigger>
          <TabsTrigger value="explainability" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <Lightbulb className="w-4 h-4 mr-1 sm:mr-2" />
            <span className="hidden sm:inline">Explainability</span>
            <span className="sm:hidden">XAI</span>
          </TabsTrigger>
          <TabsTrigger value="features" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <Filter className="w-4 h-4 mr-1 sm:mr-2" />
            Features
          </TabsTrigger>

          <TabsTrigger value="history" className="data-[state=active]:bg-teal-500 data-[state=active]:text-white text-xs sm:text-sm">
            <BarChart3 className="w-4 h-4 mr-1 sm:mr-2" />
            History
          </TabsTrigger>
        </TabsList>

        {/* Prediction Tab */}
        <TabsContent value="prediction" className="mt-6">
          <div className="grid lg:grid-cols-2 gap-6">
            {/* Left Column: Input */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
            >
              <InputForm
                formData={formData}
                setFormData={setFormData}
                onSubmit={handlePredict}
                isLoading={isLoading}
              />
            </motion.div>

            {/* Right Column: Results */}
            <div className="space-y-6">
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
              >
                {result ? (
                  <ResultsPanel result={result} />
                ) : (
                  <div className="h-64 flex items-center justify-center bg-slate-800/30 rounded-xl border border-slate-700/50">
                    <div className="text-center text-slate-400">
                      <Activity className="w-16 h-16 mx-auto mb-4 opacity-30" />
                      <p className="text-lg">Enter clinical parameters and click<br /><span className="text-teal-400 font-semibold">Predict Risk Level</span></p>
                    </div>
                  </div>
                )}
              </motion.div>

              {result && (
                <>
                  <ConfidenceGauge confidence={result.confidence} />
                  <ClinicalRecommendations riskLevel={result.riskLevel} />
                </>
              )}
            </div>
          </div>

          {/* Additional Panels after prediction */}
          {result && (
            <div className="mt-6 grid lg:grid-cols-2 gap-6">
              <PopulationContext currentRiskScore={result.score} />
              <ExportReport formData={formData} result={result} />
            </div>
          )}
        </TabsContent>

        {/* Analysis Tab */}
        <TabsContent value="analysis" className="mt-6 space-y-6">
          <WhatIfSimulator initialData={formData} />
          <RiskFactorPanel formData={formData} />
          <DoctorAnnotations predictionResult={result} />
        </TabsContent>

        {/* Models Tab */}
        <TabsContent value="models" className="mt-6">
          <ModelComparison />
        </TabsContent>

        {/* Explainability Tab */}
        <TabsContent value="explainability" className="mt-6">
          <ExplainabilityViz formData={result ? formData : null} />
        </TabsContent>

        {/* Features Tab */}
        <TabsContent value="features" className="mt-6">
          <FeatureSelection />
        </TabsContent>

        {/* History Tab */}
        <TabsContent value="history" className="mt-6">
          <PredictionsTable />
        </TabsContent>
      </Tabs>
    </div>
  );
}