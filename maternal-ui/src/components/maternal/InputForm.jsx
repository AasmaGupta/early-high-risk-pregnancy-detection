import React from 'react';
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Heart, 
  Thermometer, 
  Activity, 
  Droplets,
  User,
  Gauge,
  Loader2,
  Sparkles
} from 'lucide-react';
import { motion } from 'framer-motion';

const inputConfig = [
  { key: 'age', label: 'Age', icon: User, min: 10, max: 70, unit: 'years', color: 'from-violet-500 to-purple-500' },
  { key: 'systolicBP', label: 'Systolic BP', icon: Activity, min: 70, max: 200, unit: 'mmHg', color: 'from-rose-500 to-red-500' },
  { key: 'diastolicBP', label: 'Diastolic BP', icon: Gauge, min: 40, max: 130, unit: 'mmHg', color: 'from-orange-500 to-amber-500' },
  { key: 'bloodSugar', label: 'Blood Sugar', icon: Droplets, min: 50, max: 300, unit: 'mg/dL', color: 'from-blue-500 to-cyan-500' },
  { key: 'bodyTemp', label: 'Body Temperature', icon: Thermometer, min: 95, max: 104, unit: '°F', step: 0.1, color: 'from-teal-500 to-emerald-500' },
  { key: 'heartRate', label: 'Heart Rate', icon: Heart, min: 40, max: 120, unit: 'bpm', color: 'from-pink-500 to-rose-500' },
];

export default function InputForm({ formData, setFormData, onSubmit, isLoading, variant = 'default' }) {
  const handleSliderChange = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: value[0] }));
  };

  const handleInputChange = (key, value) => {
    const config = inputConfig.find(c => c.key === key);
    const numValue = parseFloat(value);
    if (!isNaN(numValue) && numValue >= config.min && numValue <= config.max) {
      setFormData(prev => ({ ...prev, [key]: numValue }));
    }
  };

  const isPatient = variant === 'patient';

  return (
    <Card className={`
      border-0 shadow-2xl overflow-hidden
      ${isPatient 
        ? 'bg-white' 
        : 'bg-slate-800/50 backdrop-blur-xl border border-white/10'}
    `}>
      <CardHeader className={`
        ${isPatient 
          ? 'bg-gradient-to-r from-teal-500 to-cyan-500' 
          : 'bg-gradient-to-r from-slate-700/50 to-slate-800/50 border-b border-white/10'}
      `}>
        <CardTitle className={`flex items-center gap-3 ${isPatient ? 'text-white' : 'text-white'}`}>
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${isPatient ? 'bg-white/20' : 'bg-teal-500/20'}`}>
            <Activity className={`w-5 h-5 ${isPatient ? 'text-white' : 'text-teal-400'}`} />
          </div>
          Clinical Input Parameters
        </CardTitle>
      </CardHeader>
      <CardContent className="p-6 space-y-6">
        {inputConfig.map((config, index) => (
          <motion.div
            key={config.key}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.05 }}
            className="space-y-3"
          >
            <div className="flex items-center justify-between">
              <Label className={`flex items-center gap-2 font-medium ${isPatient ? 'text-slate-700' : 'text-slate-300'}`}>
                <div className={`w-8 h-8 rounded-lg bg-gradient-to-br ${config.color} flex items-center justify-center`}>
                  <config.icon className="w-4 h-4 text-white" />
                </div>
                {config.label}
              </Label>
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={formData[config.key]}
                  onChange={(e) => handleInputChange(config.key, e.target.value)}
                  className={`w-20 h-9 text-center font-mono font-semibold
                    ${isPatient 
                      ? 'bg-slate-100 border-slate-200 text-slate-800' 
                      : 'bg-slate-700/50 border-slate-600 text-white'}
                  `}
                  min={config.min}
                  max={config.max}
                  step={config.step || 1}
                />
                <span className={`text-sm w-14 ${isPatient ? 'text-slate-500' : 'text-slate-400'}`}>{config.unit}</span>
              </div>
            </div>
            <div className="px-1">
              <Slider
                value={[formData[config.key]]}
                onValueChange={(value) => handleSliderChange(config.key, value)}
                min={config.min}
                max={config.max}
                step={config.step || 1}
                className={`cursor-pointer ${isPatient ? '' : '[&_[role=slider]]:bg-white [&_[role=slider]]:border-white [&_.relative]:bg-slate-600 [&_[data-orientation=horizontal]>.bg-primary]:bg-teal-400'}`}
              />
              <div className={`flex justify-between text-xs mt-1 ${isPatient ? 'text-slate-400' : 'text-slate-400'}`}>
                <span>{config.min}</span>
                <span>{config.max}</span>
              </div>
            </div>
          </motion.div>
        ))}

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <Button
            onClick={onSubmit}
            disabled={isLoading}
            className={`
              w-full h-14 text-lg font-semibold rounded-xl transition-all duration-300
              ${isPatient
                ? 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-600 hover:to-cyan-600 text-white shadow-lg shadow-teal-500/30'
                : 'bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-white shadow-lg shadow-teal-500/20'}
            `}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5 mr-2" />
                Predict Risk Level
              </>
            )}
          </Button>
        </motion.div>
      </CardContent>
    </Card>
  );
}