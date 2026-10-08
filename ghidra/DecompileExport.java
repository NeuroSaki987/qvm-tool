// Apply qvm-tool recovered symbols, then decompile and report.
//
// Task 3's payload: turn the recovered structure into reviewer-facing C. Reads a
// symbols TSV emitted by `qvmtool symbols`, applies names/comments in the program,
// then decompiles a selected set of functions (exports, VM entries, VM stubs) and
// records per-function decompiler success over the whole image.
//
// @category ACE.QVM

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.symbol.SourceType;

public class DecompileExport extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        String symsPath = args.length > 0 ? args[0] : null;
        String outDir = args.length > 1 ? args[1] : ".";
        long vmLo = args.length > 2 ? Long.decode(args[2]) : 0L;
        long vmHi = args.length > 3 ? Long.decode(args[3]) : 0L;
        int maxDecomp = args.length > 4 ? Integer.parseInt(args[4]) : 60;

        new File(outDir).mkdirs();
        int applied = 0, notApplied = 0;

        // ---- apply recovered symbols ----
        if (symsPath != null && new File(symsPath).exists()) {
            try (BufferedReader br = new BufferedReader(new FileReader(symsPath))) {
                String line = br.readLine();          // header
                while ((line = br.readLine()) != null) {
                    String[] f = line.split("\t");
                    if (f.length < 3) {
                        continue;
                    }
                    long addr;
                    try {
                        addr = Long.decode(f[0]);
                    } catch (NumberFormatException e) {
                        continue;
                    }
                    // The symbols TSV holds RVAs (that is what qvmtool emits project-wide),
                    // but Ghidra addresses are absolute. Feeding an RVA straight to toAddr()
                    // silently addressed memory below the image, so NOT A SINGLE NAME FROM
                    // 5,028 APPLIED -- and because the run still reported "applied 5028",
                    // the failure looked like success. Rebase anything that cannot be a VA.
                    long imageBase = currentProgram.getImageBase().getOffset();
                    if (addr < imageBase) {
                        addr += imageBase;
                    }
                    String name = f[1];
                    String kind = f[2];
                    String detail = f.length > 3 ? f[3] : "";
                    Address a = toAddr(addr);
                    Function fn = getFunctionAt(a);
                    try {
                        if (fn == null && ("function".equals(kind) || "vm_entry".equals(kind)
                                || "export".equals(kind))) {
                            fn = createFunction(a, name);
                        }
                        if (fn != null) {
                            fn.setName(name, SourceType.USER_DEFINED);
                        }
                        createLabel(a, name, true, SourceType.USER_DEFINED);
                        setPlateComment(a, detail);
                        // Count only what actually landed, by reading it back. The previous
                        // unconditional increment reported "applied 5028" while zero names
                        // were present in the program -- a counter that cannot fail is not a
                        // check, and it turned a total failure into an apparent success.
                        String landed = null;
                        if (getFunctionAt(a) != null) {
                            landed = getFunctionAt(a).getName();
                        } else if (getSymbolAt(a) != null) {
                            landed = getSymbolAt(a).getName();
                        }
                        if (name.equals(landed)) {
                            applied++;
                        } else {
                            notApplied++;
                        }
                    } catch (Exception e) {
                        notApplied++;
                    }
                }
            }
        }
        println("DecompileExport: applied " + applied + " symbols, NOT applied "
                + notApplied + " (verified by read-back)");

        // ---- decompiler health over the whole image ----
        DecompInterface di = new DecompInterface();
        di.openProgram(currentProgram);

        int total = 0, ok = 0, failed = 0, empty = 0;
        List<String> targets = new ArrayList<>();
        FunctionIterator fns = currentProgram.getFunctionManager().getFunctions(true);
        while (fns.hasNext() && !monitor.isCancelled()) {
            Function f = fns.next();
            total++;
            String name = f.getName();
            boolean interesting = name.startsWith("ace_") || name.startsWith("qvm_")
                    || name.contains("CreateObject") || name.contains("ord7");
            if (interesting) {
                targets.add(f.getEntryPoint().toString());
            }
        }
        println("DecompileExport: " + total + " functions, "
                + targets.size() + " symbolised targets");

        int written = 0;
        try (PrintWriter idx = new PrintWriter(new File(outDir, "decomp_index.tsv"))) {
            idx.println("addr\tname\tstatus\tdecompiled_lines");
            FunctionIterator all = currentProgram.getFunctionManager().getFunctions(true);
            while (all.hasNext() && !monitor.isCancelled()) {
                Function f = all.next();
                DecompileResults r = di.decompileFunction(f, 60, monitor);
                String status;
                String c = null;
                if (r == null) {
                    status = "null";
                    failed++;
                } else if (!r.decompileCompleted()) {
                    status = "incomplete";
                    failed++;
                } else {
                    c = r.getDecompiledFunction().getC();
                    if (c == null || c.trim().isEmpty()) {
                        status = "empty";
                        empty++;
                    } else {
                        status = "ok";
                        ok++;
                    }
                }
                int lines = (c == null) ? 0 : c.split("\n").length;
                idx.println(f.getEntryPoint() + "\t" + f.getName() + "\t" + status
                        + "\t" + lines);

                if (c != null && written < maxDecomp && isInteresting(f)) {
                    String safe = f.getName().replaceAll("[^A-Za-z0-9_.]", "_");
                    File out = new File(outDir, String.format("%s_%s.c",
                            f.getEntryPoint(), safe));
                    try (PrintWriter pw = new PrintWriter(out)) {
                        pw.println("// " + f.getName() + " @ " + f.getEntryPoint()
                                + "  size=" + f.getBody().getNumAddresses());
                        pw.println(c);
                    }
                    written++;
                }
            }
        }
        di.dispose();

        try (PrintWriter pw = new PrintWriter(new File(outDir, "decomp_summary.txt"))) {
            pw.println("program=" + currentProgram.getName());
            pw.println("symbols_applied=" + applied);
            pw.println("functions_total=" + total);
            pw.println("decompile_ok=" + ok);
            pw.println("decompile_failed=" + failed);
            pw.println("decompile_empty=" + empty);
            pw.println("decompiled_files_written=" + written);
            pw.println("vm_range=" + Long.toHexString(vmLo) + ".." + Long.toHexString(vmHi));
        }
        println("DecompileExport: ok=" + ok + " failed=" + failed + " empty=" + empty
                + " files=" + written);
    }

    private boolean isInteresting(Function f) {
        String n = f.getName();
        return n.startsWith("ace_") || n.startsWith("qvm_") || n.contains("CreateObject")
                || n.contains("ord7");
    }
}
