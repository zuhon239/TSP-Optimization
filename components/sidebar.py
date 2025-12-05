"""
Streamlit Sidebar Components - OPTIMIZED VERSION
Clean UI for algorithm parameter configuration
Author: Quân (Frontend Specialist)
"""
import streamlit as st
from typing import Dict, Any

def render_sidebar() -> Dict[str, Any]:
    """
    Render optimized sidebar with algorithm configuration
    
    Returns:
        Dictionary containing algorithm choice and parameters
    """
    st.sidebar.header("🔧 Algorithm Configuration")
    
    # ========================
    # ALGORITHM SELECTION
    # ========================
    algorithm_choice = st.sidebar.selectbox(
        "🎯 Select Algorithm",
        ["Genetic Algorithm (GA)", "Particle Swarm Optimization (PSO)"],
        help="Choose optimization algorithm to solve TSP"
    )
    
    st.sidebar.markdown("---")
    
    # ========================
    # ALGORITHM-SPECIFIC PARAMETERS
    # ========================
    
    if "Genetic" in algorithm_choice:
        st.sidebar.subheader("🧬 GA Parameters")
        
        with st.sidebar.expander("📊 Core Parameters", expanded=True):
            population_size = st.slider(
                "Population Size",
                min_value=20,
                max_value=200,
                value=50,
                step=10,
                help="Number of solutions in each generation"
            )
            
            generations = st.slider(
                "Generations",
                min_value=50,
                max_value=500,
                value=100,
                step=25,
                help="Number of evolution iterations"
            )
        
        with st.sidebar.expander("🔀 Genetic Operators", expanded=False):
            crossover_prob = st.slider(
                "Crossover Rate",
                min_value=0.5,
                max_value=1.0,
                value=0.8,
                step=0.05,
                help="Probability of combining two parents"
            )
            
            mutation_prob = st.slider(
                "Mutation Rate",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
                help="Probability of random gene changes"
            )
            
            tournament_size = st.slider(
                "Tournament Size",
                min_value=2,
                max_value=10,
                value=3,
                step=1,
                help="Number of individuals in tournament selection"
            )
            
            elite_size = st.slider(
                "Elite Size",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
                help="Number of best solutions preserved each generation"
            )
        
        ga_params = {
            'population_size': population_size,
            'generations': generations,
            'crossover_probability': crossover_prob,
            'mutation_probability': mutation_prob,
            'tournament_size': tournament_size,
            'elite_size': elite_size
        }
        
        # Display summary
        st.sidebar.info(f"""
        **Configuration Summary:**
        - Population: {population_size}
        - Generations: {generations}
        - Crossover: {crossover_prob:.2f}
        - Mutation: {mutation_prob:.3f}
        """)
        
        return {
            'algorithm_choice': algorithm_choice,
            'ga_params': ga_params,
            'pso_params': None
        }
    
    else:  # PSO
        st.sidebar.subheader("🐝 PSO Parameters")
        
        with st.sidebar.expander("📊 Core Parameters", expanded=True):
            swarm_size = st.slider(
                "Swarm Size",
                min_value=10,
                max_value=100,
                value=30,
                step=5,
                help="Number of particles in swarm"
            )
            
            max_iterations = st.slider(
                "Iterations",
                min_value=25,
                max_value=500,
                value=100,
                step=25,
                help="Maximum number of iterations"
            )
        
        with st.sidebar.expander("⚙️ PSO Coefficients", expanded=False):
            w = st.slider(
                "Inertia Weight (w)",
                min_value=0.1,
                max_value=1.5,
                value=0.729,
                step=0.05,
                help="Controls exploration vs exploitation"
            )
            
            c1 = st.slider(
                "Cognitive Coefficient (c1)",
                min_value=0.5,
                max_value=3.0,
                value=1.494,
                step=0.1,
                help="Particle's tendency to return to best personal position"
            )
            
            c2 = st.slider(
                "Social Coefficient (c2)",
                min_value=0.5,
                max_value=3.0,
                value=1.494,
                step=0.1,
                help="Particle's tendency to move to best global position"
            )
        
        with st.sidebar.expander("🎚️ Adaptive Inertia", expanded=False):
            w_min = st.slider(
                "Min Inertia (w_min)",
                min_value=0.1,
                max_value=0.8,
                value=0.4,
                step=0.05,
                help="Minimum inertia weight for adaptive PSO"
            )
            
            w_max = st.slider(
                "Max Inertia (w_max)",
                min_value=0.5,
                max_value=1.5,
                value=0.9,
                step=0.05,
                help="Maximum inertia weight for adaptive PSO"
            )
        
        pso_params = {
            'swarm_size': swarm_size,
            'max_iterations': max_iterations,
            'w': w,
            'c1': c1,
            'c2': c2,
            'w_min': w_min,
            'w_max': w_max
        }
        
        # Display summary
        st.sidebar.info(f"""
        **Configuration Summary:**
        - Swarm Size: {swarm_size}
        - Iterations: {max_iterations}
        - Inertia (w): {w:.3f}
        - Cognitive (c1): {c1:.2f}
        - Social (c2): {c2:.2f}
        """)
        
        return {
            'algorithm_choice': algorithm_choice,
            'ga_params': None,
            'pso_params': pso_params
        }
    
st.sidebar.markdown("---")

# ========================
# RESET BUTTON
# ========================
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Default values work well for most cases")