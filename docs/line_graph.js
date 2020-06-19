
const lineGraphMain = () => {

    const data_location = './visualization_data/best_results.json';

    const svg = d3.select('#line-graph-svg');
    const lineGraphGroup = svg.append('g');
    const lineGraphTitle = lineGraphGroup.append('text');
    const linesGroup = lineGraphGroup.append('g');
    const attentionModelLines = linesGroup.append('g');
    const plainRNNModelLines = linesGroup.append('g');
    const xAxisGroup = lineGraphGroup.append('g');
    const xAxisLabel = xAxisGroup.append('text');
    const yAxisGroup = lineGraphGroup.append('g');
    const yAxisLabel = yAxisGroup.append('text');
    const toolTipGroup = svg.append('g');
    const toolTipBoundingBox = toolTipGroup
          .append('rect')
          .style('opacity', 0);

    const margin = {
        top: 80,
        bottom: 80,
        left: 120,
        right: 30,
    };

    const attentionFill = 'red';
    const attentionToolTipFill = '#ff99ad';
    const plainRNNFill = 'blue';
    const plainRNNToolTipFill = '#9cb3ff';
    const toolTipTransitionTime = 250;
    
    const render = (attention_data, plain_rnn_data) => {
        
        const plotContainer = document.getElementById('line-graph');
        svg
            .attr('width', plotContainer.clientWidth)
            .attr('height', plotContainer.clientHeight);
        
        const svgWidth = parseFloat(svg.attr('width'));
        const svgHeight = parseFloat(svg.attr('height'));
        
        const innerWidth = svgWidth - margin.left - margin.right;
        const innerHeight = svgHeight - margin.top - margin.bottom;
        
        const labelSize = Math.min(20, innerWidth/40);
        lineGraphTitle
            .style('font-size', labelSize)
            .attr('x', innerWidth / 2)
            .attr('y', margin.top - labelSize * 2)
            .text('Validation Performance');
        
        const nameForAttentionModel = datum => `Attention Model #${datum.model_index}`;
        const nameForPlainRNNModel = datum => `RNN Model #${datum.model_index}`;
        const attentionNames = attention_data.map(nameForAttentionModel);
        const plainRNNNames = plain_rnn_data.map(nameForPlainRNNModel);
        const modelNames = attentionNames.concat(plainRNNNames);
        const modelMaxEpoch = data => Math.max(...data.validation_progress.map(datum => datum.epoch));
        const xMaxValue = Math.max(d3.max(attention_data, modelMaxEpoch), d3.max(plain_rnn_data, modelMaxEpoch));
        const xScale = d3.scaleLinear()
              .domain([0, xMaxValue])
              .range([0, innerWidth]);
        
        const yScale = d3.scaleLinear()
              .domain([1.0, 0.4])
              .range([0, innerHeight]);
        
        yAxisGroup.call(d3.axisLeft(yScale).tickSize(-innerWidth))
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        yAxisGroup.selectAll('.tick line')
            .attr('x', margin.left - 10)
            .style('opacity', 0.1);
        yAxisGroup.selectAll('.tick text')
            .attr('transform', 'translate(-10.0, 0.0)');
        yAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('transform', 'rotate(-90)')
            .attr('y', -60)
            .attr('x', -innerHeight/3)
            .text('Mean Cross Entropy Loss');
        
        const tickTextSize = Math.min(15, innerWidth/40);
        const mean = arr => arr.reduce((a,b) => a + b, 0) / arr.length;
        const tickTextAverageLength = mean(modelNames.map(name => name.length));
        xAxisGroup.call(d3.axisBottom(xScale).tickSize(-innerHeight).ticks(5))
            .attr('transform', `translate(${margin.left}, ${margin.top+innerHeight})`);
        xAxisGroup.selectAll('.tick line')
            .style('opacity', 0.1);
        xAxisGroup.selectAll('.tick text')
            .attr('transform', `translate(0, 10)`);
        xAxisLabel
            .style('font-size', labelSize * 0.8)
            .attr('fill', 'black')
            .attr('y', margin.bottom * 0.75)
            .attr('x', innerWidth / 2)
            .text('Number of Completed Epochs');

        attentionModelLines
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        plainRNNModelLines
            .attr('transform', `translate(${margin.left}, ${margin.top})`);
        
        const lineGenerator = d3.line()
              .curve(d3.curveCardinal.tension(0.7))
              .x(datum => xScale(datum.epoch))
              .y(datum => yScale(datum.mean_loss));
        
        const updateToolTip = (mouseX, mouseY, desiredOpacity, model, backgroundColor) => {
            toolTipGroup.selectAll('g').remove();
            const toolTipTextLines = [
                `Number of Epochs: ${model.number_of_epochs}`,
                `Batch Size: ${model.batch_size}`,
                `Vocab Size: ${model.vocab_size}`,
                `Pretrained Embedding: ${model.pre_trained_embedding_specification}`,
                `LSTM Hidden Size: ${model.encoding_hidden_size}`,
                `Number of LSTM Layers: ${model.number_of_encoding_layers}`
            ];
            if (model.attention_intermediate_size) {
                toolTipTextLines.push(
                    `Attention Intermediate Size: ${model.attention_intermediate_size}`,
                    `Number of Attention Heads: ${model.number_of_attention_heads}`
                );
            }
            toolTipTextLines.push(
                `Dropout Probability: ${model.dropout_probability}`,
                `Test Loss: ${model.test_loss}`,
                `Test Accuracy: ${model.test_accuracy}`,
                `Number of Parameters: ${model.number_of_parameters}`
            );
            const ephemeralTextLinesGroup = toolTipGroup.append('g');
            toolTipTextLines.forEach((textLine, textLineIndex) => {
                ephemeralTextLinesGroup
                    .append('text')
                    .style('font-size', labelSize)
                    .attr('class', 'displayed-text')
                    .attr('dx', labelSize)
                    .attr('dy', `${(1+textLineIndex) * 1.2}em`)
                    .html(textLine);
            });
            const ephemeralTextLinesGroupBBox = ephemeralTextLinesGroup.node().getBBox();
            const toolTipBoundingBoxWidth = ephemeralTextLinesGroupBBox.width + 2 * labelSize;
            const toolTipBoundingBoxHeight = ephemeralTextLinesGroupBBox.height + labelSize;
            
            const rightLimit = margin.left + innerWidth;
            const mouseCloserToRight = mouseX - margin.left > rightLimit - mouseX;
            const toolTipX = desiredOpacity === 0 ? -svgWidth : (mouseCloserToRight ? margin.left + labelSize : rightLimit - labelSize - toolTipBoundingBoxWidth);
            
            const bottomLimit = margin.top + innerHeight;
            const mouseCloserToBottom = mouseY - margin.top > bottomLimit - mouseY;
            const toolTipY = desiredOpacity === 0 ? -svgHeight : (mouseCloserToBottom ? margin.top + labelSize : bottomLimit - labelSize - toolTipBoundingBoxHeight);
            console.log('\n\n\n');
            console.log(`mouseY - margin.top ${JSON.stringify(mouseY - margin.top)}`);
            console.log(`bottomLimit - mouseY ${JSON.stringify(bottomLimit - mouseY)}`);
            console.log(`mouseCloserToBottom ${JSON.stringify(mouseCloserToBottom)}`);
            console.log(`toolTipY ${JSON.stringify(toolTipY)}`);
            
            toolTipBoundingBox
                .attr('x', toolTipX)
                .attr('y', toolTipY)
                .style('stroke-width', 1)
                .style('stroke', 'black')
                .style('fill', backgroundColor)
                .attr('width', toolTipBoundingBoxWidth)
                .attr('height', toolTipBoundingBoxHeight);
            ephemeralTextLinesGroup.selectAll('*')
                .attr('x', toolTipX)
                .attr('y', toolTipY);
            const elementsSelection = toolTipGroup.selectAll('*');
            elementsSelection
                .transition()
                .duration(toolTipTransitionTime)
                .style('opacity', desiredOpacity);
        };
        
        attentionModelLines.selectAll('path')
            .remove();
        attention_data.forEach(model => {
            attentionModelLines
                .append('path')
                .style('stroke-opacity', 0.5)
                .style('stroke-width', 2)
                .style('transition', 'all 0.25s')
                .on('mouseover', function() {
                    const [mouseX, mouseY] = d3.mouse(this);
                    updateToolTip(mouseX, mouseY, 1, model, attentionToolTipFill);
                    linesGroup
                        .selectAll('path')
                        .style('stroke-width', 1)
                        .style('stroke-opacity', 0.25);
                    d3.select(this)
                        .style('stroke-width', 4)
                        .style('stroke-opacity', 1);
                })
                .on('mouseout', function() {
                    const [mouseX, mouseY] = d3.mouse(this);
                    updateToolTip(mouseX, mouseY, 0, model, attentionToolTipFill);
                    linesGroup
                        .selectAll('path')
                        .style('stroke-width', 2)
                        .style('stroke-opacity', 0.5);
                })
                .attr('class', 'path')
                .attr('d', lineGenerator(model.validation_progress))
                .attr('fill', 'none')
                .attr('stroke', attentionFill);
        });

        plainRNNModelLines.selectAll('path')
            .remove();
        plain_rnn_data.forEach(model => {
            plainRNNModelLines
                .append('path')
                .style('stroke-opacity', 0.5)
                .style('stroke-width', 2)
                .style('transition', 'all 0.25s')
                .on('mouseover', function() {
                    const [mouseX, mouseY] = d3.mouse(this);
                    updateToolTip(mouseX, mouseY, 1, model, plainRNNToolTipFill);
                    linesGroup
                        .selectAll('path')
                        .style('stroke-width', 1)
                        .style('stroke-opacity', 0.25);
                    d3.select(this)
                        .style('stroke-width', 4)
                        .style('stroke-opacity', 1);
                })
                .on('mouseout', function() {
                    const [mouseX, mouseY] = d3.mouse(this);
                    updateToolTip(mouseX, mouseY, 0, model, plainRNNToolTipFill);
                    linesGroup
                        .selectAll('path')
                        .style('stroke-width', 2)
                        .style('stroke-opacity', 0.5);
                })
                .attr('class', 'path')
                .attr('d', lineGenerator(model.validation_progress))
                .attr('fill', 'none')
                .attr('stroke', plainRNNFill);
        });
                 
    };
    
    d3.json(data_location)
        .then(data => {
            let extract_data = (datum, datumIndex) => {
                return {
                    model_index: datumIndex, 
                    number_of_epochs: parseInt(datum.number_of_epochs),
                    batch_size: parseInt(datum.batch_size),
                    vocab_size: parseInt(datum.vocab_size),
                    pre_trained_embedding_specification: datum.pre_trained_embedding_specification,
                    encoding_hidden_size: parseInt(datum.encoding_hidden_size),
                    number_of_encoding_layers: parseInt(datum.number_of_encoding_layers),
                    attention_intermediate_size: parseInt(datum.attention_intermediate_size),
                    number_of_attention_heads: parseInt(datum.number_of_attention_heads),
                    dropout_probability: parseFloat(datum.dropout_probability),
                    final_representation: datum.final_representation,
                    test_loss: parseFloat(datum.test_loss),
                    test_accuracy: parseFloat(datum.test_accuracy),
                    number_of_parameters: parseInt(datum.number_of_parameters),
                    training_progress: [{epoch: 0, mean_loss: 1.0}].concat(datum.training_progress),
                    validation_progress: [{epoch: 0, mean_loss: 1.0}].concat(datum.validation_progress),
                };
            };
            let attention_data = data['attention'].map(extract_data);
            let plain_rnn_data = data['plain_rnn'].map(extract_data);
            const redraw = () => {
                render(attention_data, plain_rnn_data);
            };
            redraw();
            window.addEventListener('resize', redraw);
        }).catch(err => {
            console.error(err.message);
            return;
        });

};

lineGraphMain();
