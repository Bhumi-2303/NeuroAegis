import { useState, useRef } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Upload, FileText, User, Activity, Brain, ShieldCheck } from 'lucide-react';
import { motion } from 'framer-motion';

const patientSchema = z.object({
  name: z.string().regex(/^[a-zA-Z\s]+$/, "Only alphabets allowed").min(2, "Name required"),
  age: z.coerce.number().min(0).max(120),
  gender: z.enum(["Male", "Female", "Other"]),
  weight: z.coerce.number().min(20).max(250),
  height: z.coerce.number().min(40).max(250),
  medical_history: z.string().min(5, "Medical history required"),
  heart_rate: z.coerce.number().min(30).max(200),
  blood_pressure: z.string().regex(/^\d{2,3}\/\d{2,3}$/, "Format: 120/80"),
  spo2: z.coerce.number().min(50).max(100),
  temperature: z.coerce.number().min(30).max(45),
  consent_given: z.literal(true, { errorMap: () => ({ message: "You must provide informed consent to proceed." }) }),
});

type PatientFormData = z.infer<typeof patientSchema>;

export function PatientForm({ onStartAnalysis }: { onStartAnalysis: (jobId: string) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const { register, handleSubmit, formState: { errors } } = useForm<PatientFormData>({
    resolver: zodResolver(patientSchema) as any,
    defaultValues: {
      name: "John Doe",
      age: 35,
      gender: "Male",
      weight: 70,
      height: 170,
      medical_history: "No prior seizure history",
      heart_rate: 75,
      blood_pressure: "120/80",
      spo2: 98,
      temperature: 36.6
    }
  });

  const onSubmit = async (data: PatientFormData) => {
    setFormError(null);
    if (!file) {
      setFormError("Please upload an EEG file before submitting.");
      return;
    }

    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append('name', data.name);
      formData.append('age', data.age.toString());
      formData.append('gender', data.gender);
      formData.append('weight', data.weight.toString());
      formData.append('height', data.height.toString());
      
      const medicalHistory = { notes: data.medical_history };
      formData.append('medical_history', JSON.stringify(medicalHistory));
      
      const vitalSigns = {
        heart_rate: data.heart_rate,
        blood_pressure: data.blood_pressure,
        spo2: data.spo2,
        temperature: data.temperature
      };
      formData.append('vital_signs', JSON.stringify(vitalSigns));
      
      formData.append('file', file);
      formData.append('sampling_rate', '256');

      const response = await fetch('/api/v2/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to start prediction");
      }

      const result = await response.json();
      onStartAnalysis(result.job_id);

    } catch (e: any) {
      console.error(e);
      setFormError(e.message || "Error starting analysis. Please check backend connection.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
      <div className="lg:col-span-2 space-y-6">
        <form id="patient-form" onSubmit={handleSubmit(onSubmit)} className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-xl">
          <div className="flex items-center gap-3 mb-8 pb-4 border-b border-white/10">
            <User className="text-teal-400 w-6 h-6" />
            <h2 className="text-2xl font-semibold">Patient Demographics</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">Full Name</label>
              <input {...register("name")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" placeholder="John Doe" />
              {errors.name && <p className="text-red-400 text-xs mt-1">{errors.name.message}</p>}
            </div>
            
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">Age</label>
              <input type="number" {...register("age")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
              {errors.age && <p className="text-red-400 text-xs mt-1">{errors.age.message}</p>}
            </div>

            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">Gender</label>
              <select {...register("gender")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors">
                <option value="">Select...</option>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
                <option value="Other">Other</option>
              </select>
              {errors.gender && <p className="text-red-400 text-xs mt-1">{errors.gender.message}</p>}
            </div>

            <div className="flex gap-4">
              <div className="flex-1">
                <label className="block text-sm font-medium text-neutral-400 mb-2">Weight (kg)</label>
                <input type="number" {...register("weight")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
                {errors.weight && <p className="text-red-400 text-xs mt-1">{errors.weight.message}</p>}
              </div>
              <div className="flex-1">
                <label className="block text-sm font-medium text-neutral-400 mb-2">Height (cm)</label>
                <input type="number" {...register("height")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
                {errors.height && <p className="text-red-400 text-xs mt-1">{errors.height.message}</p>}
              </div>
            </div>
          </div>

          <div className="mt-8 mb-8 pb-4 border-b border-white/10 flex items-center gap-3">
            <Activity className="text-blue-400 w-6 h-6" />
            <h2 className="text-2xl font-semibold">Vital Signs</h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">Heart Rate</label>
              <input type="number" {...register("heart_rate")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
              {errors.heart_rate && <p className="text-red-400 text-xs mt-1">{errors.heart_rate.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">BP</label>
              <input {...register("blood_pressure")} placeholder="120/80" className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
              {errors.blood_pressure && <p className="text-red-400 text-xs mt-1">{errors.blood_pressure.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">SpO2 (%)</label>
              <input type="number" {...register("spo2")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
              {errors.spo2 && <p className="text-red-400 text-xs mt-1">{errors.spo2.message}</p>}
            </div>
            <div>
              <label className="block text-sm font-medium text-neutral-400 mb-2">Temp (°C)</label>
              <input type="number" step="0.1" {...register("temperature")} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors" />
              {errors.temperature && <p className="text-red-400 text-xs mt-1">{errors.temperature.message}</p>}
            </div>
          </div>

          <div className="mt-8">
            <label className="block text-sm font-medium text-neutral-400 mb-2">Medical History & Symptoms</label>
            <textarea {...register("medical_history")} rows={4} className="w-full bg-black/40 border border-white/10 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-teal-500 transition-colors resize-none" placeholder="Previous seizures, medication, family history..."></textarea>
            {errors.medical_history && <p className="text-red-400 text-xs mt-1">{errors.medical_history.message}</p>}
          </div>

          {/* Informed Consent */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <div className="flex items-center gap-3 mb-4">
              <ShieldCheck className="text-amber-400 w-5 h-5" />
              <h3 className="text-lg font-semibold text-amber-200">Informed Consent</h3>
            </div>
            <div className="bg-amber-500/5 border border-amber-500/20 rounded-xl p-4 mb-4">
              <p className="text-sm text-neutral-300 leading-relaxed">
                By checking the box below, I confirm that:
              </p>
              <ul className="text-sm text-neutral-400 list-disc list-inside mt-2 space-y-1">
                <li>I have the legal authority to submit this patient's data for analysis.</li>
                <li>I understand this tool is a <strong className="text-amber-300">research aid only</strong> and is NOT a certified medical device.</li>
                <li>All clinical decisions must be made by a qualified medical professional.</li>
                <li>Patient data will be processed per our <a href="/privacy" className="text-teal-400 underline hover:text-teal-300">Privacy Policy</a>.</li>
                <li>I may request deletion of this data at any time by contacting an administrator.</li>
              </ul>
            </div>
            <label className="flex items-start gap-3 cursor-pointer group">
              <input
                type="checkbox"
                {...register("consent_given")}
                className="mt-1 w-5 h-5 rounded border-2 border-amber-500/40 bg-black/40 checked:bg-amber-500 checked:border-amber-500 focus:ring-amber-500/50 focus:ring-2 transition-all cursor-pointer"
              />
              <span className="text-sm text-neutral-300 group-hover:text-white transition-colors">
                I have read and agree to the above terms and give informed consent for this analysis.
              </span>
            </label>
            {errors.consent_given && <p className="text-red-400 text-xs mt-2 ml-8">{errors.consent_given.message}</p>}
          </div>
        </form>
      </div>

      <div className="space-y-6">
        <div className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-xl h-full flex flex-col">
          <div className="flex items-center gap-3 mb-6 pb-4 border-b border-white/10">
            <Brain className="text-purple-400 w-6 h-6" />
            <h2 className="text-2xl font-semibold">EEG Data</h2>
          </div>
          
          <div 
            onClick={() => fileInputRef.current?.click()}
            className={`flex-1 border-2 border-dashed ${file ? 'border-teal-500/50 bg-teal-500/10' : 'border-white/20 bg-black/40 hover:border-white/40'} rounded-xl flex flex-col items-center justify-center p-8 cursor-pointer transition-all group`}
          >
            <input 
              type="file" 
              accept=".csv,.edf,.json,.txt" 
              ref={fileInputRef}
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            
            {file ? (
              <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex flex-col items-center text-center">
                <FileText className="w-12 h-12 text-teal-400 mb-4" />
                <p className="font-medium text-teal-100">{file.name}</p>
                <p className="text-xs text-teal-500/70 mt-2">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                <p className="text-xs text-teal-400 mt-4 cursor-pointer hover:underline">Click to change file</p>
              </motion.div>
            ) : (
              <div className="flex flex-col items-center text-center opacity-70 group-hover:opacity-100 transition-opacity">
                <Upload className="w-12 h-12 mb-4" />
                <p className="font-medium mb-2">Upload EEG Recording</p>
                <p className="text-sm text-neutral-400">Supports .csv, .edf, .json, .txt</p>
                <p className="text-xs text-neutral-500 mt-4">Drag & drop or click to browse</p>
              </div>
            )}
          </div>
          
          {!file && <p className="text-red-400 text-xs mt-4 text-center">An EEG file is required to proceed</p>}
          {formError && <p className="text-red-400 text-xs mt-2 text-center font-medium bg-red-500/10 p-2 rounded-lg border border-red-500/20">{formError}</p>}

          <button 
            type="submit" 
            form="patient-form"
            disabled={!file || isSubmitting}
            className="mt-8 w-full bg-gradient-to-r from-teal-500 to-blue-600 hover:from-teal-400 hover:to-blue-500 text-white font-semibold py-4 rounded-xl shadow-lg shadow-teal-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-[1.02] active:scale-95"
          >
            {isSubmitting ? "Initializing..." : "Analyze Patient"}
          </button>
        </div>
      </div>
    </div>
  );
}
