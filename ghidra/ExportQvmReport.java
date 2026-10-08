// Export a function/CFG report from Ghidra headless.
//
// Reports, per function: address, name, size, instruction count, basic-block count,
// call count, whether Ghidra recovered any decompilable body, and the incoming
// reference count. Also dumps a summary block and the top-level call graph edges
// that land inside the VM section, so the original and deobfuscated builds can be
// compared numerically.
//
// @category ACE.QVM

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.block.BasicBlockModel;
import ghidra.program.model.block.CodeBlock;
import ghidra.program.model.block.CodeBlockIterator;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionIterator;
import ghidra.program.model.listing.Instruction;
import ghidra.program.model.listing.InstructionIterator;
import ghidra.program.model.symbol.Reference;
import ghidra.program.model.symbol.ReferenceIterator;

public class ExportQvmReport extends GhidraScript {

    @Override
    protected void run() throws Exception {
        String[] args = getScriptArgs();
        String outPath = (args.length > 0) ? args[0] : "qvm_report.tsv";
        long vmLo = (args.length > 1) ? Long.decode(args[1]) : 0L;
        long vmHi = (args.length > 2) ? Long.decode(args[2]) : 0L;

        BasicBlockModel bbm = new BasicBlockModel(currentProgram);
        FunctionIterator fns = currentProgram.getFunctionManager().getFunctions(true);

        List<String> rows = new ArrayList<>();
        int totalFunctions = 0;
        int noBody = 0;
        long totalInstrs = 0;
        long totalBlocks = 0;
        int vmFunctions = 0;
        int vmIncomingEdges = 0;

        while (fns.hasNext() && !monitor.isCancelled()) {
            Function f = fns.next();
            totalFunctions++;
            Address entry = f.getEntryPoint();
            long addr = entry.getOffset();

            if (f.getBody() == null) {
                noBody++;
                continue;
            }

            long insnCount = 0;
            InstructionIterator ii = currentProgram.getListing().getInstructions(f.getBody(), true);
            while (ii.hasNext()) {
                ii.next();
                insnCount++;
            }
            totalInstrs += insnCount;

            int blocks = 0;
            try {
                CodeBlockIterator bi = bbm.getCodeBlocksContaining(f.getBody(), monitor);
                while (bi.hasNext()) {
                    bi.next();
                    blocks++;
                }
            } catch (Exception e) {
                // block model can fail on malformed bodies; leave blocks at 0
            }
            totalBlocks += blocks;

            int inRefs = 0;
            ReferenceIterator ri = currentProgram.getReferenceManager()
                    .getReferencesTo(entry);
            while (ri.hasNext()) {
                Reference r = ri.next();
                inRefs++;
                Address from = r.getFromAddress();
                if (from != null) {
                    long fromOff = from.getOffset();
                    if (vmLo != 0 && fromOff >= vmLo && fromOff < vmHi) {
                        vmIncomingEdges++;
                    }
                }
            }

            boolean inVm = (vmLo != 0 && addr >= vmLo && addr < vmHi);
            if (inVm) {
                vmFunctions++;
            }

            rows.add(String.format("%d\t%s\t%d\t%d\t%d\t%d\t%s",
                    addr, f.getName(), f.getBody().getNumAddresses(),
                    insnCount, blocks, inRefs, inVm ? "VM" : "native"));
        }

        File out = new File(outPath);
        if (out.getParentFile() != null) {
            out.getParentFile().mkdirs();
        }
        try (PrintWriter pw = new PrintWriter(out)) {
            pw.println("# program=" + currentProgram.getName());
            pw.println("# imageBase=" + currentProgram.getImageBase());
            pw.println("# totalFunctions=" + totalFunctions);
            pw.println("# functionsWithBody=" + (totalFunctions - noBody));
            pw.println("# functionsWithoutBody=" + noBody);
            pw.println("# totalInstructions=" + totalInstrs);
            pw.println("# totalBasicBlocks=" + totalBlocks);
            pw.println("# vmFunctions=" + vmFunctions);
            pw.println("# vmIncomingEdges=" + vmIncomingEdges);
            pw.println("# addr\tname\tsize\tinstructions\tblocks\tinRefs\tkind");
            for (String r : rows) {
                pw.println(r);
            }
        }
        println("ExportQvmReport: wrote " + out.getAbsolutePath()
                + " (" + rows.size() + " functions)");
    }
}
