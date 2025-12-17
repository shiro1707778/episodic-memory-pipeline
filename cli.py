#!/usr/bin/env python3
"""
Command-line interface for the episodic memory pipeline.

Usage:
    python cli.py ingest "I started learning Korean today"
    python cli.py query "What am I learning?"
    python cli.py recall "Tell me about my Korean learning journey"
    python cli.py consolidate --topic language_learning
    python cli.py stats
    python cli.py demo
    python cli.py eval --scenario diary
    python cli.py doctor  # Diagnostic command

All initialization (including FAISS/SentenceTransformers ordering) is handled
by the bootstrap module. See src/bootstrap.py for details.
"""
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

# Import bootstrap - this handles all the initialization ordering
from src.bootstrap import get_components, PipelineComponents
from config import config

console = Console()

# Global cache for components (initialized on first command)
_components: PipelineComponents = None


def get_pipeline_components(use_mock: bool = False) -> PipelineComponents:
    """
    Get or create pipeline components via bootstrap.
    
    Args:
        use_mock: Force mock providers for testing
        
    Returns:
        PipelineComponents with all initialized components
    """
    global _components
    
    # Only initialize once, but respect use_mock if specified
    if _components is None or use_mock:
        _components = get_components(config=config, force_mock=use_mock, verbose=True)
    
    return _components


@click.group()
@click.option('--mock', is_flag=True, help='Use mock providers (no API calls)')
@click.pass_context
def cli(ctx, mock):
    """Episodic Memory Pipeline CLI"""
    ctx.ensure_object(dict)
    ctx.obj['use_mock'] = mock


@cli.command()
@click.argument('text')
@click.option('--source', default='cli', help='Source of the input')
@click.option('--force', is_flag=True, help='Skip worthiness check')
@click.pass_context
def ingest(ctx, text, source, force):
    """Ingest a piece of text into memory."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    pipeline = c.IngestionPipeline(
        c.database, c.vector_store, c.embedding_provider, c.llm,
        worthiness_threshold=config.memory_worthiness_threshold
    )
    
    with console.status("Processing input..."):
        result = pipeline.ingest(text, source=source, force=force)
    
    if result.success:
        console.print(Panel(
            f"[green]✓ Memory stored successfully[/green]\n\n"
            f"[bold]Content:[/bold] {result.episode.content}\n"
            f"[bold]Type:[/bold] {result.episode.memory_type}\n"
            f"[bold]Topics:[/bold] {', '.join(result.episode.topics) or 'none'}\n"
            f"[bold]Importance:[/bold] {result.episode.importance:.2f}\n"
            f"[bold]ID:[/bold] {result.episode.id[:8]}",
            title="Ingestion Result"
        ))
    else:
        console.print(Panel(
            f"[yellow]⊘ Not stored[/yellow]\n\n"
            f"[bold]Reason:[/bold] {result.reason}\n"
            f"[bold]Confidence:[/bold] {result.classification.confidence:.2f if result.classification else 'N/A'}",
            title="Ingestion Result"
        ))


@cli.command()
@click.argument('query_text')
@click.option('--no-synthesize', is_flag=True, help='Skip answer synthesis')
@click.pass_context
def query(ctx, query_text, no_synthesize):
    """Query the memory system."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    engine = c.RetrievalEngine(c.database, c.vector_store, c.embedding_provider, c.llm)
    
    with console.status("Searching memories..."):
        result = engine.query(query_text, synthesize=not no_synthesize)
    
    # Display answer
    if result.answer:
        console.print(Panel(
            Markdown(result.answer),
            title=f"Answer (confidence: {result.confidence:.1%})",
            border_style="green" if result.confidence > 0.7 else "yellow"
        ))
    
    # Display supporting evidence
    if result.facts:
        console.print("\n[bold]Related Facts:[/bold]")
        for fact in result.facts[:5]:
            console.print(f"  • {fact.content} [dim](conf: {fact.confidence:.1%})[/dim]")
    
    if result.episodes:
        console.print(f"\n[bold]Supporting Episodes:[/bold] ({len(result.episodes)} found)")
        for ep in result.episodes[:5]:
            date_str = ep.occurred_at.strftime("%Y-%m-%d")
            console.print(f"  • [{date_str}] {ep.content[:80]}...")
    
    if result.gaps:
        console.print(f"\n[dim]Gaps: {', '.join(result.gaps)}[/dim]")


@cli.command()
@click.argument('topic_or_query')
@click.option('--topic', is_flag=True, help='Treat input as exact topic name')
@click.pass_context
def recall(ctx, topic_or_query, topic):
    """Recall the narrative/journey for a topic."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    engine = c.RetrievalEngine(c.database, c.vector_store, c.embedding_provider, c.llm)
    
    with console.status("Recalling narrative..."):
        result = engine.recall_narrative(topic_or_query, is_topic=topic)
    
    # Display narrative
    console.print(Panel(
        Markdown(result.answer),
        title=f"Narrative: {topic_or_query}",
        border_style="blue"
    ))
    
    # Display timeline
    if result.episodes:
        console.print("\n[bold]Timeline:[/bold]")
        for ep in result.episodes[:10]:
            date_str = ep.occurred_at.strftime("%Y-%m-%d %H:%M")
            console.print(f"  [{date_str}] {ep.content[:60]}...")


@cli.command()
@click.option('--topic', help='Consolidate specific topic')
@click.option('--all', 'consolidate_all', is_flag=True, help='Consolidate all topics needing it')
@click.pass_context
def consolidate(ctx, topic, consolidate_all):
    """Run memory consolidation."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    pipeline = c.ConsolidationPipeline(
        c.database, c.vector_store, c.embedding_provider, c.llm,
        episode_threshold=config.consolidation_episode_threshold,
        age_threshold_days=config.consolidation_age_days
    )
    
    if topic:
        with console.status(f"Consolidating topic: {topic}..."):
            result = pipeline.consolidate_topic(topic)
        results = [result]
    elif consolidate_all:
        with console.status("Consolidating all topics..."):
            results = pipeline.consolidate_all()
    else:
        console.print("[red]Please specify --topic or --all[/red]")
        return
    
    if not results:
        console.print("[yellow]No topics needed consolidation[/yellow]")
        return
    
    # Display results
    table = Table(title="Consolidation Results")
    table.add_column("Topic")
    table.add_column("Episodes")
    table.add_column("Summaries")
    table.add_column("Facts")
    table.add_column("Duration")
    
    for r in results:
        table.add_row(
            r.topic or "all",
            str(r.episodes_processed),
            str(r.summaries_created),
            str(r.facts_extracted),
            f"{r.duration_seconds:.2f}s"
        )
    
    console.print(table)


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory system statistics."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    db_stats = c.database.get_statistics()
    vec_stats = c.vector_store.get_statistics()
    
    # Database stats
    table = Table(title="Database Statistics")
    table.add_column("Metric")
    table.add_column("Value")
    
    table.add_row("Total Episodes", str(db_stats["total_episodes"]))
    table.add_row("Unconsolidated Episodes", str(db_stats["unconsolidated_episodes"]))
    table.add_row("Total Facts", str(db_stats["total_facts"]))
    table.add_row("Total Summaries", str(db_stats["total_summaries"]))
    table.add_row("Total Topics", str(db_stats["total_topics"]))
    
    console.print(table)
    
    # Vector stats
    table2 = Table(title="Vector Store Statistics")
    table2.add_column("Index")
    table2.add_column("Count")
    table2.add_column("Dimension")
    
    for name, info in vec_stats.items():
        table2.add_row(name, str(info["count"]), str(info["dimension"]))
    
    console.print(table2)
    
    # Topics
    topics = c.database.get_topics()
    if topics:
        table3 = Table(title="Topics")
        table3.add_column("Name")
        table3.add_column("Episodes")
        table3.add_column("Last Consolidated")
        
        for t in topics[:10]:
            last_cons = t["last_consolidation"] or "never"
            table3.add_row(t["name"], str(t["episode_count"]), str(last_cons))
        
        console.print(table3)


@cli.command()
@click.pass_context
def demo(ctx):
    """Run an interactive demo."""
    console.print(Panel(
        "[bold]Episodic Memory Pipeline Demo[/bold]\n\n"
        "This demo will walk through the core functionality:\n"
        "1. Ingesting memories\n"
        "2. Running consolidation\n"
        "3. Querying memories\n"
        "4. Narrative recall",
        title="Welcome"
    ))
    
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    ingestion = c.IngestionPipeline(
        c.database, c.vector_store, c.embedding_provider, c.llm,
        worthiness_threshold=config.memory_worthiness_threshold
    )
    
    consolidation = c.ConsolidationPipeline(
        c.database, c.vector_store, c.embedding_provider, c.llm
    )
    
    retrieval = c.RetrievalEngine(c.database, c.vector_store, c.embedding_provider, c.llm)
    
    # Demo memories
    demo_memories = [
        "I started learning Korean today. My goal is to be conversational by March for my Seoul trip.",
        "I've been practicing Korean for 2 hours. Learned basic greetings: 안녕하세요, 감사합니다.",
        "My friend recommended the Talk To Me In Korean podcast. Going to try it tomorrow.",
        "Had my first conversation in Korean today! Just basic stuff but it felt great.",
        "I prefer visual learning over audio. Going to focus more on writing practice.",
        "Booked my flight to Seoul for March 15th. Excited but nervous about the language barrier.",
    ]
    
    # Ingest demo memories
    console.print("\n[bold cyan]Step 1: Ingesting demo memories...[/bold cyan]\n")
    
    for text in demo_memories:
        result = ingestion.ingest(text, source="demo", force=True)
        status = "[green]✓[/green]" if result.success else "[red]✗[/red]"
        console.print(f"  {status} {text[:50]}...")
    
    console.print("\n[bold cyan]Step 2: Running consolidation...[/bold cyan]\n")
    
    results = consolidation.consolidate_all()
    if results:
        for r in results:
            console.print(f"  Consolidated '{r.topic}': {r.episodes_processed} episodes → {r.summaries_created} summary, {r.facts_extracted} facts")
    else:
        console.print("  No topics needed consolidation")
    
    console.print("\n[bold cyan]Step 3: Semantic query...[/bold cyan]\n")
    
    query_text = "What am I learning right now?"
    console.print(f"  Query: \"{query_text}\"")
    result = retrieval.query(query_text)
    console.print(Panel(result.answer, title="Answer"))
    
    console.print("\n[bold cyan]Step 4: Narrative recall...[/bold cyan]\n")
    
    recall_text = "Tell me about my Korean learning journey"
    console.print(f"  Query: \"{recall_text}\"")
    result = retrieval.recall_narrative("korean", is_topic=False)
    console.print(Panel(result.answer, title="Narrative"))
    
    # Final stats
    console.print("\n[bold cyan]Final Statistics:[/bold cyan]\n")
    ctx.invoke(stats)
    
    console.print("\n[green]Demo complete![/green]")


@cli.command()
@click.pass_context
def interactive(ctx):
    """Start an interactive session."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    ingestion = c.IngestionPipeline(
        c.database, c.vector_store, c.embedding_provider, c.llm,
        worthiness_threshold=config.memory_worthiness_threshold
    )
    
    retrieval = c.RetrievalEngine(c.database, c.vector_store, c.embedding_provider, c.llm)
    
    console.print(Panel(
        "[bold]Interactive Memory Session[/bold]\n\n"
        "Commands:\n"
        "  /remember <text> - Store a memory\n"
        "  /query <text>    - Query memories\n"
        "  /recall <topic>  - Recall narrative\n"
        "  /stats           - Show statistics\n"
        "  /quit            - Exit\n\n"
        "Or just type naturally - it will be analyzed for memory-worthiness.",
        title="Welcome"
    ))
    
    while True:
        try:
            user_input = console.input("\n[bold]>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['/quit', '/exit', '/q']:
            console.print("[dim]Goodbye![/dim]")
            break
        
        if user_input.startswith('/remember '):
            text = user_input[10:]
            result = ingestion.ingest(text, source="interactive", force=True)
            if result.success:
                console.print(f"[green]✓ Remembered:[/green] {result.episode.content}")
            else:
                console.print(f"[yellow]Not stored:[/yellow] {result.reason}")
        
        elif user_input.startswith('/query '):
            query_text = user_input[7:]
            result = retrieval.query(query_text)
            console.print(Panel(result.answer, title="Answer"))
        
        elif user_input.startswith('/recall '):
            topic = user_input[8:]
            result = retrieval.recall_narrative(topic, is_topic=True)
            console.print(Panel(result.answer, title=f"Narrative: {topic}"))
        
        elif user_input == '/stats':
            db_stats = c.database.get_statistics()
            console.print(f"Episodes: {db_stats['total_episodes']} | Facts: {db_stats['total_facts']} | Summaries: {db_stats['total_summaries']}")
        
        else:
            # Try to ingest as memory
            result = ingestion.ingest(user_input, source="interactive")
            if result.success:
                console.print(f"[green]✓ Noted:[/green] {result.episode.content[:50]}...")
            else:
                console.print(f"[dim]({result.reason})[/dim]")


@cli.command()
@click.option('--scenario', '-s', default='diary', help='Evaluation scenario to run')
@click.option('--k', default=5, help='K value for precision@k metric')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed output')
@click.pass_context
def eval(ctx, scenario, k, verbose):
    """Run evaluation metrics on the memory pipeline."""
    c = get_pipeline_components(ctx.obj['use_mock'])
    
    console.print(Panel(
        f"[bold]Episodic Memory Pipeline Evaluation[/bold]\n\n"
        f"Scenario: {scenario}\n"
        f"Precision@K: {k}\n"
        f"This will create an isolated test environment.",
        title="Evaluation"
    ))
    
    # Get scenario
    try:
        eval_scenario = c.get_scenario(scenario)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        return
    
    console.print(f"\n[dim]{eval_scenario.description}[/dim]\n")
    
    # Create runner
    runner = c.EvaluationRunner(
        embedding_provider=c.embedding_provider,
        llm=c.llm,
        precision_k=k,
    )
    
    # Run evaluation
    with console.status(f"Running {scenario} scenario..."):
        result = runner.run_scenario(eval_scenario)
    
    if not result.success:
        console.print(f"[red]Evaluation failed: {result.error}[/red]")
        return
    
    # Display results
    console.print(f"\n[green]✓ Evaluation completed in {result.duration_seconds:.2f}s[/green]\n")
    
    # Summary table
    table = Table(title="Evaluation Summary")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Details")
    
    metrics = result.metrics
    
    # Counts
    table.add_row("Episodes Ingested", str(metrics.episode_count), "")
    table.add_row("Facts Extracted", str(metrics.fact_count), "")
    table.add_row("Summaries Created", str(metrics.summary_count), "")
    
    console.print(table)
    
    # Show warnings for mock providers
    if metrics.using_mock_embeddings:
        console.print("[yellow]⚠ MOCK EMBEDDINGS: Retrieval metrics are not meaningful[/yellow]")
    if metrics.using_mock_llm:
        console.print("[yellow]⚠ MOCK LLM: Fact extraction metrics may not be meaningful[/yellow]")
    console.print()
    
    # Retrieval metrics
    if metrics.retrieval_precision:
        rp = metrics.retrieval_precision
        
        if metrics.using_mock_embeddings:
            table2 = Table(title="Retrieval Metrics [dim](SKIPPED - mock embeddings)[/dim]")
            table2.add_column("Metric")
            table2.add_column("Value")
            table2.add_row(f"Precision@{rp.k}", "[dim]SKIPPED[/dim]")
            table2.add_row("Recall", "[dim]SKIPPED[/dim]")
            table2.add_row("F1 Score", "[dim]SKIPPED[/dim]")
            table2.add_row("Note", "[dim]Mock embeddings have no semantic meaning[/dim]")
        else:
            table2 = Table(title="Retrieval Metrics")
            table2.add_column("Metric")
            table2.add_column("Value")
            
            precision_color = "green" if rp.precision_at_k >= 0.6 else "yellow" if rp.precision_at_k >= 0.4 else "red"
            table2.add_row(
                f"Precision@{rp.k}",
                f"[{precision_color}]{rp.precision_at_k:.1%}[/{precision_color}]"
            )
            table2.add_row("Recall", f"{rp.recall:.1%}")
            table2.add_row("F1 Score", f"{rp.f1:.1%}")
            table2.add_row("Relevant Found", f"{rp.relevant_found}/{rp.total_expected}")
        
        console.print(table2)
    
    # Fact consistency metrics
    if metrics.fact_conflict:
        fc = metrics.fact_conflict
        
        table3 = Table(title="Fact Consistency Metrics")
        table3.add_column("Metric")
        table3.add_column("Value")
        
        consistency_color = "green" if fc.consistency_rate >= 0.9 else "yellow" if fc.consistency_rate >= 0.7 else "red"
        table3.add_row(
            "Consistency Rate",
            f"[{consistency_color}]{fc.consistency_rate:.1%}[/{consistency_color}]"
        )
        table3.add_row("Conflict Rate", f"{fc.conflict_rate:.1%}")
        table3.add_row("Conflicting Facts", f"{fc.conflicting_facts}/{fc.total_facts}")
        
        console.print(table3)
        
        if verbose and fc.conflict_pairs:
            console.print("\n[bold]Conflict Details:[/bold]")
            for f1_id, f2_id, reason in fc.conflict_pairs[:5]:
                console.print(f"  • {f1_id[:8]} ↔ {f2_id[:8]}: {reason}")
    
    # Compression metrics
    if metrics.compression:
        cm = metrics.compression
        
        table4 = Table(title="Consolidation Compression")
        table4.add_column("Metric")
        table4.add_column("Value")
        
        # Good compression is 0.1-0.3 (70-90% reduction)
        ratio_color = "green" if 0.1 <= cm.compression_ratio <= 0.4 else "yellow"
        if cm.compression_ratio == 0:
            ratio_color = "dim"
            ratio_display = "N/A (no summaries)"
        else:
            ratio_display = f"{cm.compression_ratio:.2f}"
        
        table4.add_row("Compression Ratio", f"[{ratio_color}]{ratio_display}[/{ratio_color}]")
        table4.add_row("Source Tokens", str(cm.source_tokens))
        table4.add_row("Summary Tokens", str(cm.summary_tokens))
        
        if cm.compression_ratio > 0:
            reduction = (1 - cm.compression_ratio) * 100
            table4.add_row("Size Reduction", f"{reduction:.0f}%")
        
        console.print(table4)
    
    # Overall assessment
    console.print("\n" + "─" * 50)
    
    overall_score = _compute_overall_score(metrics)
    score_color = "green" if overall_score >= 0.7 else "yellow" if overall_score >= 0.5 else "red"
    
    console.print(f"\n[bold]Overall Score: [{score_color}]{overall_score:.1%}[/{score_color}][/bold]")
    
    # Interpretation
    if overall_score >= 0.7:
        console.print("[green]✓ Memory pipeline is performing well[/green]")
    elif overall_score >= 0.5:
        console.print("[yellow]⚠ Memory pipeline has room for improvement[/yellow]")
    else:
        console.print("[red]✗ Memory pipeline needs attention[/red]")


def _compute_overall_score(metrics) -> float:
    """Compute weighted overall score from metrics."""
    scores = []
    weights = []
    
    if metrics.retrieval_precision:
        # Precision is most important
        scores.append(metrics.retrieval_precision.precision_at_k)
        weights.append(0.4)
        
        # F1 provides balanced view
        scores.append(metrics.retrieval_precision.f1)
        weights.append(0.2)
    
    if metrics.fact_conflict:
        # Consistency matters
        scores.append(metrics.fact_conflict.consistency_rate)
        weights.append(0.25)
    
    if metrics.compression and metrics.compression.compression_ratio > 0:
        # Good compression is 0.1-0.3, map to 0-1 score
        ratio = metrics.compression.compression_ratio
        # Score peaks at 0.2 ratio
        compression_score = max(0, 1 - abs(ratio - 0.2) * 3)
        scores.append(compression_score)
        weights.append(0.15)
    
    if not scores:
        return 0.0
    
    # Normalize weights
    total_weight = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_weight


# =============================================================================
# DOCTOR COMMAND - Diagnostic/Debugging Tool
# =============================================================================

@cli.command()
@click.option('--dry', is_flag=True, help='Dry-run mode: inspect config only, no initialization')
@click.pass_context
def doctor(ctx, dry):
    """
    Run system diagnostics and show configuration status.
    
    This command inspects configuration, provider selection, and bootstrap state
    without making any LLM calls or modifying data. Use it to:
    
    - Debug configuration issues
    - Verify provider selection before running evaluations
    - Understand why metrics might be zero or skipped
    - Review system state for code reviews
    
    Use --dry for a safe config-only inspection that doesn't initialize any
    models, FAISS indices, or database connections.
    """
    from src.diagnostics import (
        get_config_diagnostics,
        describe_llm_provider,
        describe_embedding_provider,
        generate_fix_suggestions,
        format_status_icon,
        format_bool_display,
        format_env_value,
    )
    
    use_mock = ctx.obj.get('use_mock', False)
    
    if dry:
        _doctor_dry_run(use_mock)
    else:
        _doctor_full(ctx, use_mock)


def _doctor_dry_run(use_mock: bool):
    """
    Run doctor in dry-run mode: inspect config without initializing components.
    
    This is safe to run even if dependencies are missing or misconfigured.
    """
    import os
    from src.diagnostics import (
        get_config_diagnostics,
        generate_fix_suggestions,
        format_status_icon,
        format_env_value,
    )
    
    console.print(Panel(
        "[bold]Episodic Memory Pipeline - System Diagnostics[/bold]\n"
        "[yellow]DRY RUN — no components initialized[/yellow]",
        title="Doctor (Dry Run)",
        border_style="yellow"
    ))
    
    # Get config diagnostics without initializing anything
    diag = get_config_diagnostics(config, force_mock=use_mock)
    
    # =========================================================================
    # Section 1: Environment Variables
    # =========================================================================
    table1 = Table(title="Environment Variables", show_header=True, header_style="bold cyan")
    table1.add_column("Variable", style="dim")
    table1.add_column("Value")
    table1.add_column("Effect")
    
    table1.add_row(
        "EMBEDDING_PROVIDER",
        format_env_value(diag.env_embedding_provider),
        f"→ {diag.resolved_embedding_provider}"
    )
    table1.add_row(
        "EMBEDDING_MODEL",
        format_env_value(diag.env_embedding_model),
        f"→ {diag.resolved_embedding_model}"
    )
    table1.add_row(
        "EMBEDDING_DEVICE",
        format_env_value(diag.env_embedding_device),
        f"→ {diag.resolved_embedding_device}"
    )
    table1.add_row(
        "EMBEDDING_DIMENSION",
        format_env_value(diag.env_embedding_dimension),
        f"→ {diag.resolved_embedding_dimension}"
    )
    table1.add_row(
        "LLM_PROVIDER",
        format_env_value(diag.env_llm_provider),
        f"→ {diag.resolved_llm_provider}"
    )
    table1.add_row(
        "OLLAMA_MODEL",
        format_env_value(diag.env_ollama_model),
        f"→ {config.ollama_model}" if diag.resolved_llm_provider == "ollama" else "[dim]N/A[/dim]"
    )
    table1.add_row(
        "OPENAI_API_KEY",
        "[green]set[/green]" if diag.env_openai_api_key_set else "[dim]not set[/dim]",
        "Required for OpenAI provider"
    )
    table1.add_row(
        "TOKENIZERS_PARALLELISM",
        format_env_value(diag.env_tokenizers_parallelism),
        "[green]safe[/green]" if diag.env_tokenizers_parallelism == "false" else "[yellow]should be 'false'[/yellow]"
    )
    
    console.print(table1)
    console.print()
    
    # =========================================================================
    # Section 2: Provider Selection Preview
    # =========================================================================
    table2 = Table(title="Provider Selection (Predicted)", show_header=True, header_style="bold cyan")
    table2.add_column("Component", style="dim")
    table2.add_column("Will Use")
    table2.add_column("Status")
    
    # Embedding prediction
    if use_mock:
        emb_status = "[yellow]⚠ MOCK (--mock flag)[/yellow]"
    elif diag.will_use_mock_embeddings:
        emb_status = "[yellow]⚠ MOCK[/yellow]"
    else:
        emb_status = "[green]✓ Real[/green]"
    
    table2.add_row(
        "Embeddings",
        f"{diag.resolved_embedding_provider}" if not diag.will_use_mock_embeddings else "mock",
        emb_status
    )
    
    # LLM prediction
    if use_mock:
        llm_status = "[yellow]⚠ MOCK (--mock flag)[/yellow]"
    elif diag.will_use_mock_llm:
        llm_status = "[yellow]⚠ MOCK (no API key)[/yellow]"
    else:
        llm_status = "[green]✓ Real[/green]"
    
    table2.add_row(
        "LLM",
        f"{diag.resolved_llm_provider}" if not diag.will_use_mock_llm else "mock",
        llm_status
    )
    
    console.print(table2)
    console.print()
    
    # =========================================================================
    # Section 3: Evaluation Readiness (Predicted)
    # =========================================================================
    table3 = Table(title="Evaluation Readiness (Predicted)", show_header=True, header_style="bold cyan")
    table3.add_column("Check", style="dim")
    table3.add_column("Status")
    table3.add_column("Impact")
    
    if diag.will_use_mock_embeddings:
        table3.add_row(
            "Embeddings",
            "[yellow]⚠ Will be MOCK[/yellow]",
            "Retrieval metrics will be SKIPPED"
        )
    else:
        table3.add_row(
            "Embeddings",
            "[green]✓ Will be Real[/green]",
            f"Using {diag.resolved_embedding_provider}"
        )
    
    if diag.will_use_mock_llm:
        table3.add_row(
            "LLM",
            "[yellow]⚠ Will be MOCK[/yellow]",
            "Fact/consolidation metrics may not be meaningful"
        )
    else:
        table3.add_row(
            "LLM",
            "[green]✓ Will be Real[/green]",
            f"Using {diag.resolved_llm_provider}"
        )
    
    console.print(table3)
    console.print()
    
    # =========================================================================
    # Section 4: Suggested Fixes
    # =========================================================================
    suggestions = generate_fix_suggestions(config_diag=diag, force_mock=use_mock)
    
    if suggestions:
        suggestion_text = "\n".join(suggestions)
        console.print(Panel(
            f"[bold]Copy-paste these commands to fix issues:[/bold]\n\n"
            f"[cyan]{suggestion_text}[/cyan]",
            title="Suggested Fixes",
            border_style="blue"
        ))
    
    console.print()
    console.print("[dim]Run without --dry to see full diagnostics with initialized components.[/dim]")


def _doctor_full(ctx, use_mock: bool):
    """
    Run doctor with full component initialization.
    
    This is the original doctor behavior that initializes all components.
    """
    import os
    from src.bootstrap import is_initialized, get_cached_embedding_model
    from src.diagnostics import (
        describe_llm_provider,
        describe_embedding_provider,
        generate_fix_suggestions,
        format_status_icon,
        format_bool_display,
    )
    
    console.print(Panel(
        "[bold]Episodic Memory Pipeline - System Diagnostics[/bold]",
        title="Doctor",
        border_style="blue"
    ))
    
    # Get components (this will initialize if not already)
    c = get_pipeline_components(use_mock)
    
    # =========================================================================
    # Section 1: Bootstrap Status
    # =========================================================================
    table1 = Table(title="Bootstrap Status", show_header=True, header_style="bold cyan")
    table1.add_column("Check", style="dim")
    table1.add_column("Status")
    table1.add_column("Details")
    
    # Bootstrap initialized
    bootstrap_init = is_initialized()
    table1.add_row(
        "Bootstrap initialized",
        format_status_icon(bootstrap_init),
        "FAISS/SentenceTransformers init order enforced" if bootstrap_init else "Not using bootstrap"
    )
    
    # Embedding model preloaded
    cached_model = get_cached_embedding_model()
    has_cached = cached_model is not None
    table1.add_row(
        "Embedding model preloaded",
        format_status_icon(has_cached),
        f"Model cached in memory" if has_cached else "No preloaded model"
    )
    
    # TOKENIZERS_PARALLELISM
    tokenizers_disabled = os.environ.get('TOKENIZERS_PARALLELISM', '').lower() == 'false'
    table1.add_row(
        "TOKENIZERS_PARALLELISM",
        format_status_icon(tokenizers_disabled, warning_if_false=True),
        "false (safe)" if tokenizers_disabled else "not set (may cause issues)"
    )
    
    # FAISS safe import
    table1.add_row(
        "FAISS safe-import order",
        "[green]ENFORCED[/green]" if bootstrap_init else "[yellow]UNKNOWN[/yellow]",
        "Bootstrap module manages import ordering"
    )
    
    console.print(table1)
    console.print()
    
    # =========================================================================
    # Section 2: LLM Provider
    # =========================================================================
    table2 = Table(title="LLM Provider", show_header=True, header_style="bold cyan")
    table2.add_column("Property", style="dim")
    table2.add_column("Value")
    
    llm = c.llm
    llm_info = describe_llm_provider(llm, config)
    
    table2.add_row("Provider type", llm_info.type)
    table2.add_row("Model", llm_info.model)
    table2.add_row("Temperature", llm_info.temperature)
    table2.add_row("Is mock", format_bool_display(llm_info.is_mock))
    if llm_info.base_url:
        table2.add_row("Base URL", llm_info.base_url)
    
    console.print(table2)
    console.print()
    
    # =========================================================================
    # Section 3: Embedding Provider
    # =========================================================================
    table3 = Table(title="Embedding Provider", show_header=True, header_style="bold cyan")
    table3.add_column("Property", style="dim")
    table3.add_column("Value")
    
    emb = c.embedding_provider
    emb_info = describe_embedding_provider(emb, config)
    
    table3.add_row("Provider type", emb_info.type)
    table3.add_row("Model", emb_info.model)
    table3.add_row("Device", emb_info.device)
    table3.add_row("Dimension", str(emb_info.dimension))
    table3.add_row("Normalized", format_bool_display(emb_info.normalized))
    table3.add_row("Is mock", format_bool_display(emb_info.is_mock))
    
    console.print(table3)
    console.print()
    
    # =========================================================================
    # Section 4: Vector Store / FAISS
    # =========================================================================
    table4 = Table(title="Vector Store (FAISS)", show_header=True, header_style="bold cyan")
    table4.add_column("Property", style="dim")
    table4.add_column("Value")
    
    vs = c.vector_store
    vs_stats = vs.get_statistics()
    
    # Index type - FAISS IndexFlatIP for inner product (cosine on normalized)
    table4.add_row("Index type", "IndexFlatIP (Inner Product)")
    table4.add_row("Similarity metric", "Cosine (via inner product on L2-normalized vectors)")
    table4.add_row("Index dimension", str(vs.dimension))
    
    # Dimension consistency check
    dim_match = vs.dimension == emb.dimension
    dim_status = "[green]✓ Match[/green]" if dim_match else f"[red]✗ MISMATCH (embedding={emb.dimension})[/red]"
    table4.add_row("Dimension consistency", dim_status)
    
    # Index sizes
    total_vectors = 0
    for idx_name, idx_info in vs_stats.items():
        count = idx_info.get('count', 0)
        total_vectors += count
        table4.add_row(f"  {idx_name} vectors", str(count))
    
    table4.add_row("Total vectors", str(total_vectors))
    
    console.print(table4)
    console.print()
    
    # =========================================================================
    # Section 5: Evaluation Readiness
    # =========================================================================
    table5 = Table(title="Evaluation Readiness", show_header=True, header_style="bold cyan")
    table5.add_column("Check", style="dim")
    table5.add_column("Status")
    table5.add_column("Impact")
    
    warnings = []
    
    # Mock embeddings check
    if emb_info.is_mock:
        warnings.append("mock_embeddings")
        table5.add_row(
            "Embeddings",
            "[yellow]⚠ MOCK[/yellow]",
            "Retrieval metrics will be SKIPPED"
        )
    else:
        table5.add_row(
            "Embeddings",
            "[green]✓ Real[/green]",
            f"Using {emb_info.type} ({emb_info.model})"
        )
    
    # Mock LLM check
    if llm_info.is_mock:
        warnings.append("mock_llm")
        table5.add_row(
            "LLM",
            "[yellow]⚠ MOCK[/yellow]",
            "Fact/consolidation metrics may not be meaningful"
        )
    else:
        table5.add_row(
            "LLM",
            "[green]✓ Real[/green]",
            f"Using {llm_info.type} ({llm_info.model})"
        )
    
    # Dimension consistency check
    if not dim_match:
        warnings.append("dimension_mismatch")
        table5.add_row(
            "Dimensions",
            "[red]✗ MISMATCH[/red]",
            "Vector store and embedding dimensions don't match!"
        )
    else:
        table5.add_row(
            "Dimensions",
            "[green]✓ Consistent[/green]",
            f"All using {emb.dimension}d vectors"
        )
    
    console.print(table5)
    console.print()
    
    # Overall status
    if not warnings:
        console.print(Panel(
            "[green]✓ System ready for meaningful evaluation[/green]\n\n"
            "All providers are configured with real models.\n"
            "Run `python cli.py eval --scenario diary` to test.",
            border_style="green"
        ))
    else:
        warning_text = "[yellow]⚠ System has warnings that may affect evaluation:[/yellow]\n\n"
        if "mock_embeddings" in warnings:
            warning_text += "• Mock embeddings → Retrieval precision will be SKIPPED\n"
        if "mock_llm" in warnings:
            warning_text += "• Mock LLM → Fact extraction metrics may not be meaningful\n"
        if "dimension_mismatch" in warnings:
            warning_text += "• Dimension mismatch → Vector search will fail!\n"
        
        warning_text += "\n[dim]Use real providers for meaningful evaluation results.[/dim]"
        console.print(Panel(warning_text, border_style="yellow"))
    
    # =========================================================================
    # Section 6: Suggested Fixes (new)
    # =========================================================================
    if warnings:
        suggestions = generate_fix_suggestions(llm_info=llm_info, emb_info=emb_info, force_mock=use_mock)
        if suggestions:
            suggestion_text = "\n".join(suggestions)
            console.print()
            console.print(Panel(
                f"[bold]Copy-paste these commands to fix issues:[/bold]\n\n"
                f"[cyan]{suggestion_text}[/cyan]",
                title="Suggested Fixes",
                border_style="blue"
            ))


if __name__ == "__main__":
    cli()
