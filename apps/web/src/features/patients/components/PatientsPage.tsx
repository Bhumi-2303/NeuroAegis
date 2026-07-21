import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Users, Search, Plus, User, Activity, Clock, Edit2, X, Save } from 'lucide-react';
import { GlassCard, EmptyState } from '../../../shared/components';
import { pageTransition, fadeIn, slideUp } from '../../../shared/lib/motion-presets';

// Using backend Patient schema now
interface Patient {
  id: string;
  name: string;
  age: number;
  gender: string;
  weight: number;
  height: number;
  medical_history: string;
  vital_signs?: any;
  status: 'active' | 'monitoring' | 'discharged';
  last_visit: string;
  created_at: string;
}

export const PatientsPage = () => {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [search, setSearch] = useState('');
  const [view, setView] = useState<'list' | 'detail' | 'form'>('list');
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null);

  // Form State
  const [formData, setFormData] = useState<Partial<Patient>>({});

  const fetchPatients = async () => {
    try {
      const res = await fetch('/api/v1/patients/');
      if (res.ok) {
        const data = await res.json();
        setPatients(data);
      }
    } catch (err) {
      console.error('Failed to fetch patients', err);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const filteredPatients = patients.filter(p => 
    p.name.toLowerCase().includes(search.toLowerCase()) || 
    p.id.toLowerCase().includes(search.toLowerCase())
  );

  const handleEdit = (p: Patient) => {
    setSelectedPatient(p);
    setFormData(p);
    setView('form');
  };

  const handleCreate = () => {
    setSelectedPatient(null);
    setFormData({ status: 'active' });
    setView('form');
  };

  const handleSave = async () => {
    try {
      if (selectedPatient) {
        // Edit existing patient (assuming PUT endpoint exists, or just fallback if not)
        // For now, API doesn't have PUT /patients/{id}, so we might just do a local update
        // In a real app we'd call a PUT endpoint
        setPatients(patients.map(p => p.id === selectedPatient.id ? { ...p, ...formData } as Patient : p));
      } else {
        // Create new patient
        const newPatientData = { 
          ...formData,
        };
        const res = await fetch('/api/v1/patients/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newPatientData)
        });
        
        if (res.ok) {
          const createdPatient = await res.json();
          setPatients([...patients, createdPatient]);
        }
      }
      setView('list');
    } catch (err) {
      console.error('Failed to save patient', err);
    }
  };

  return (
    <motion.div {...pageTransition} className="p-6 max-w-6xl mx-auto space-y-6">
      <header className="flex items-center justify-between mb-8">
        <div className="flex items-center gap-3">
          <Users className="w-8 h-8 text-[var(--accent-primary)]" />
          <h1 className="text-3xl font-[var(--font-display)] font-bold text-[var(--text-primary)]">
            Patient Management
          </h1>
        </div>
        
        {view === 'list' && (
          <button 
            onClick={handleCreate}
            className="flex items-center gap-2 bg-[var(--accent-primary)] text-[var(--bg-1)] px-4 py-2 rounded-lg font-[var(--font-body)] font-medium hover:opacity-90 transition-opacity"
          >
            <Plus className="w-4 h-4" /> Add Patient
          </button>
        )}
        {view !== 'list' && (
          <button 
            onClick={() => setView('list')}
            className="flex items-center gap-2 glass-card text-[var(--text-secondary)] hover:text-[var(--text-primary)] px-4 py-2 rounded-lg font-[var(--font-body)] transition-colors"
          >
            <X className="w-4 h-4" /> Cancel
          </button>
        )}
      </header>

      <AnimatePresence mode="wait">
        {view === 'list' && (
          <motion.div key="list" {...fadeIn} className="space-y-6">
            <GlassCard className="p-4 flex items-center gap-4">
              <Search className="w-5 h-5 text-[var(--text-secondary)]" />
              <input 
                type="text" 
                placeholder="Search by name or ID..."
                value={search}
                onChange={e => setSearch(e.target.value)}
                className="bg-transparent border-none outline-none w-full text-[var(--text-primary)] font-[var(--font-body)] placeholder:text-[var(--text-secondary)]"
              />
            </GlassCard>

            {filteredPatients.length === 0 ? (
              <GlassCard className="p-12">
                <EmptyState 
                  icon={Users} 
                  title="No Patients Found" 
                  description="Try adjusting your search criteria." 
                />
              </GlassCard>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {filteredPatients.map(patient => (
                  <GlassCard 
                    key={patient.id} 
                    className="p-5 hover:border-[var(--accent-primary)]/50 transition-colors cursor-pointer group"
                    onClick={() => { setSelectedPatient(patient); setView('detail'); }}
                  >
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-lg font-[var(--font-display)] font-medium text-[var(--text-primary)] group-hover:text-[var(--accent-primary)] transition-colors">
                          {patient.name}
                        </h3>
                        <span className="text-xs font-[var(--font-mono)] text-[var(--text-secondary)]">{patient.id}</span>
                      </div>
                      <div className={`w-2 h-2 rounded-full ${
                        patient.status === 'active' ? 'bg-[var(--state-danger)]' : 
                        patient.status === 'monitoring' ? 'bg-[var(--state-warning)]' : 'bg-[var(--state-success)]'
                      }`} />
                    </div>
                    
                    <div className="space-y-2 text-sm font-[var(--font-body)] text-[var(--text-secondary)]">
                      <div className="flex justify-between">
                        <span>Age/Gender:</span>
                        <span className="text-[var(--text-primary)]">{patient.age} / {patient.gender}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Last Visit:</span>
                        <span className="text-[var(--text-primary)] font-[var(--font-mono)] text-xs">
                          {new Date(patient.last_visit || patient.created_at || Date.now()).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                  </GlassCard>
                ))}
              </div>
            )}
          </motion.div>
        )}

        {view === 'detail' && selectedPatient && (
          <motion.div key="detail" {...slideUp} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <GlassCard className="p-6 lg:col-span-1">
              <div className="flex flex-col items-center text-center pb-6 border-b border-[var(--bg-3)]">
                <div className="w-20 h-20 rounded-full bg-[var(--bg-2)] flex items-center justify-center mb-4 text-[var(--accent-primary)]">
                  <User className="w-10 h-10" />
                </div>
                <h2 className="text-2xl font-[var(--font-display)] font-bold text-[var(--text-primary)]">{selectedPatient.name}</h2>
                <span className="text-sm font-[var(--font-mono)] text-[var(--text-secondary)]">{selectedPatient.id}</span>
              </div>
              
              <div className="py-6 space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-[var(--text-secondary)] text-sm font-[var(--font-body)]">Status</span>
                  <span className="capitalize text-sm text-[var(--text-primary)]">{selectedPatient.status}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[var(--text-secondary)] text-sm font-[var(--font-body)]">Age</span>
                  <span className="text-sm text-[var(--text-primary)]">{selectedPatient.age}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[var(--text-secondary)] text-sm font-[var(--font-body)]">Gender</span>
                  <span className="text-sm text-[var(--text-primary)]">{selectedPatient.gender}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[var(--text-secondary)] text-sm font-[var(--font-body)]">Weight (kg)</span>
                  <span className="text-sm text-[var(--text-primary)]">{selectedPatient.weight}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-[var(--text-secondary)] text-sm font-[var(--font-body)]">Height (cm)</span>
                  <span className="text-sm text-[var(--text-primary)]">{selectedPatient.height}</span>
                </div>
              </div>
              
              <button 
                onClick={() => handleEdit(selectedPatient)}
                className="w-full py-2 flex items-center justify-center gap-2 bg-[var(--bg-2)] hover:bg-[var(--bg-3)] text-[var(--text-primary)] rounded-lg transition-colors font-[var(--font-body)] text-sm"
              >
                <Edit2 className="w-4 h-4" /> Edit Profile
              </button>
            </GlassCard>

            <div className="lg:col-span-2 space-y-6">
              <GlassCard title="Medical History" className="p-6">
                <div className="flex gap-4">
                  <Activity className="w-5 h-5 text-[var(--accent-secondary)] shrink-0" />
                  <p className="text-[var(--text-primary)] font-[var(--font-body)] text-sm leading-relaxed">
                    {selectedPatient.medical_history || "No medical history recorded."}
                  </p>
                </div>
              </GlassCard>

              <GlassCard title="Recent Activity" className="p-6">
                 <div className="flex gap-4">
                  <Clock className="w-5 h-5 text-[var(--accent-highlight)] shrink-0" />
                  <div>
                    <h4 className="text-[var(--text-primary)] font-[var(--font-body)] text-sm font-medium mb-1">Last Clinic Visit</h4>
                    <p className="text-[var(--text-secondary)] text-sm font-[var(--font-mono)]">
                      {new Date(selectedPatient.last_visit || selectedPatient.created_at || Date.now()).toLocaleString()}
                    </p>
                  </div>
                </div>
              </GlassCard>
            </div>
          </motion.div>
        )}

        {view === 'form' && (
          <motion.div key="form" {...slideUp}>
            <GlassCard title={selectedPatient ? 'Edit Patient' : 'New Patient'} className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Full Name</label>
                  <input 
                    type="text" 
                    value={formData.name || ''}
                    onChange={e => setFormData({...formData, name: e.target.value})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  />
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Status</label>
                  <select 
                    value={formData.status || 'active'}
                    onChange={e => setFormData({...formData, status: e.target.value as any})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  >
                    <option value="active">Active</option>
                    <option value="monitoring">Monitoring</option>
                    <option value="discharged">Discharged</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Age</label>
                  <input 
                    type="number" 
                    value={formData.age || ''}
                    onChange={e => setFormData({...formData, age: parseInt(e.target.value)})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  />
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Gender</label>
                  <select 
                    value={formData.gender || ''}
                    onChange={e => setFormData({...formData, gender: e.target.value})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  >
                    <option value="">Select...</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Weight (kg)</label>
                  <input 
                    type="number" 
                    value={formData.weight || ''}
                    onChange={e => setFormData({...formData, weight: parseFloat(e.target.value)})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  />
                </div>
                <div>
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Height (cm)</label>
                  <input 
                    type="number" 
                    value={formData.height || ''}
                    onChange={e => setFormData({...formData, height: parseFloat(e.target.value)})}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-2 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)]"
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-[var(--font-body)] text-[var(--text-secondary)] mb-2">Medical History</label>
                  <textarea 
                    value={formData.medical_history || ''}
                    onChange={e => setFormData({...formData, medical_history: e.target.value})}
                    rows={4}
                    className="w-full bg-[var(--bg-2)] border border-[var(--bg-3)] rounded-lg px-4 py-3 text-sm text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-primary)] resize-none"
                  />
                </div>
              </div>
              
              <div className="flex justify-end gap-3 pt-4 border-t border-[var(--bg-3)]">
                <button 
                  onClick={() => setView('list')}
                  className="px-6 py-2 rounded-lg font-[var(--font-body)] text-sm text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-2)] transition-colors"
                >
                  Cancel
                </button>
                <button 
                  onClick={handleSave}
                  disabled={!formData.name}
                  className="px-6 py-2 rounded-lg font-[var(--font-body)] text-sm text-[var(--bg-1)] bg-[var(--accent-primary)] hover:opacity-90 disabled:opacity-50 transition-opacity flex items-center gap-2"
                >
                  <Save className="w-4 h-4" /> Save Patient
                </button>
              </div>
            </GlassCard>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};
