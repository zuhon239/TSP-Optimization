"""
Streamlit Sidebar Components - DUAL ALGORITHM CONFIGURATION
Shows parameters for both GA and PSO simultaneously
Author: Quân (Frontend Specialist)
"""
import streamlit as st
from typing import Dict, Any

def render_sidebar() -> Dict[str, Any]:
    """
    Render sidebar with both GA and PSO parameters always visible
    
    Returns:
        Dictionary containing parameters for both algorithms
    """
    st.sidebar.header("🔧 Algorithm Configuration")
    st.sidebar.caption("Configure both algorithms - Choose which to run in Step 4")
    
    st.sidebar.markdown("---")
    
    # ========================
    # GENETIC ALGORITHM PARAMETERS
    # ========================
    st.sidebar.subheader("🧬 Genetic Algorithm")
    
    with st.sidebar.expander("📊 GA Core Parameters", expanded=True):
        population_size = st.slider(
            "GA Population Size",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of solutions in each generation",
            key="ga_population"
        )
        
        generations = st.slider(
            "GA Generations",
            min_value=50,
            max_value=500,
            value=100,
            step=25,
            help="Number of evolution iterations",
            key="ga_generations"
        )
    
    with st.sidebar.expander("🔀 GA Genetic Operators", expanded=False):
        crossover_prob = st.slider(
            "GA Crossover Rate",
            min_value=0.5,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Probability of combining two parents",
            key="ga_crossover"
        )
        
        mutation_prob = st.slider(
            "GA Mutation Rate",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Probability of random gene changes",
            key="ga_mutation"
        )
        
        tournament_size = st.slider(
            "GA Tournament Size",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            help="Number of individuals in tournament selection",
            key="ga_tournament"
        )
        
        elite_size = st.slider(
            "GA Elite Size",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
            help="Number of best solutions preserved each generation",
            key="ga_elite"
        )
    
    ga_params = {
        'population_size': population_size,
        'generations': generations,
        'crossover_probability': crossover_prob,
        'mutation_probability': mutation_prob,
        'tournament_size': tournament_size,
        'elite_size': elite_size
    }
    
    st.sidebar.markdown("---")
    
    # ========================
    # PSO PARAMETERS
    # ========================
    st.sidebar.subheader("🐝 Particle Swarm Optimization")
    
    with st.sidebar.expander("📊 PSO Core Parameters", expanded=True):
        swarm_size = st.slider(
            "PSO Swarm Size",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Number of particles in swarm",
            key="pso_swarm"
        )
        
        max_iterations = st.slider(
            "PSO Max Iterations",
            min_value=50,
            max_value=500,
            value=100,
            step=25,
            help="Maximum number of iterations",
            key="pso_iterations"
        )
    
    with st.sidebar.expander("⚙️ PSO Coefficients", expanded=False):
        w = st.slider(
            "PSO Inertia Weight (w)",
            min_value=0.4,
            max_value=0.9,
            value=0.729,
            step=0.01,
            help="Controls exploration vs exploitation",
            key="pso_w"
        )
        
        c1 = st.slider(
            "PSO Cognitive Factor (c1)",
            min_value=1.0,
            max_value=2.5,
            value=1.494,
            step=0.1,
            help="Particle's tendency to return to best personal position",
            key="pso_c1"
        )
        
        c2 = st.slider(
            "PSO Social Factor (c2)",
            min_value=1.0,
            max_value=2.5,
            value=1.494,
            step=0.1,
            help="Particle's tendency to move to best global position",
            key="pso_c2"
        )
    
    with st.sidebar.expander("🎚️ PSO Adaptive Inertia", expanded=False):
        w_min = st.slider(
            "PSO Min Inertia (w_min)",
            min_value=0.1,
            max_value=0.8,
            value=0.4,
            step=0.05,
            help="Minimum inertia weight for adaptive PSO",
            key="pso_w_min"
        )
    
    pso_params = {
        'swarm_size': swarm_size,
        'max_iterations': max_iterations,
        'w': w,
        'c1': c1,
        'c2': c2,
        'w_min': w_min
    }
    
    st.sidebar.markdown("---")
    
    # ========================
    # SUMMARY INFO
    # ========================
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.info(f"""
        **🧬 GA Config:**
        - Pop: {population_size}
        - Gen: {generations}
        - Cross: {crossover_prob:.2f}
        - Mut: {mutation_prob:.3f}
        """)
    
    with col2:
        st.info(f"""
        **🐝 PSO Config:**
        - Swarm: {swarm_size}
        - Iter: {max_iterations}
        - w: {w:.3f}
        - c1: {c1:.2f}
        """)
    
    st.sidebar.markdown("---")
    
    return {
        'ga_params': ga_params,
        'pso_params': pso_params
    }